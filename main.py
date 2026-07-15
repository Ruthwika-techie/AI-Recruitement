"""
TechVest Recruitment Agent -- Main Runner
==========================================
Runs the complete LangGraph agent pipeline:

  1. Parses all three candidate resumes (+ injection detection)
  2. Scores each candidate against the JD rubric
  3. Decides: interview / hold / reject
  4. Pauses for human approval before scheduling (guardrail #1)
  5. On approval, proposes interview slots
  6. Logs the full trajectory to audit/

Usage:
    python main.py                        # full run with human gate
    python main.py --auto-approve         # skip human prompt (CI / demo mode)
    python main.py --fairness-check       # run name-swap fairness test first
"""

import argparse
import io
import json
import sys
import uuid

# Fix Windows console Unicode issues
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from dotenv import load_dotenv

load_dotenv()

from data import CANDIDATES, JD, RUBRIC
from graph import build_graph
from guardrails import fairness_ok, persist_audit

# ---------------------------------------------------------------------------
# Pretty printers
# ---------------------------------------------------------------------------

def print_banner(text: str) -> None:
    width = 70
    print("\n" + "=" * width)
    print(f"  {text}")
    print("=" * width)


def print_shortlist(shortlist: list[dict]) -> None:
    print_banner("SHORTLIST & DECISIONS")
    verdict_icon = {"interview": "[YES]", "hold": "[HOLD]", "reject": "[NO]"}
    for d in shortlist:
        icon = verdict_icon.get(d["verdict"], "?")
        print(f"\n{icon}  {d['name']}  |  score={d['weighted']:.2f}  |  verdict={d['verdict'].upper()}")
        print(f"     Justification: {d['justification']}")
        if d.get("proposed_slot"):
            print(f"     Interview slot: {d['proposed_slot']}")
        print(f"     Scorecard:")
        for cs in d["scorecard"]["scores"]:
            print(f"       [{cs['score']}/5] {cs['criterion']}: {cs['evidence'][:80]}")


def print_trajectory(trajectory: list[dict]) -> None:
    print_banner("FULL TRAJECTORY (thought -> action -> observation)")
    for i, entry in enumerate(trajectory, 1):
        print(f"\n  [{i:02d}]", end="")
        for k, v in entry.items():
            if isinstance(v, dict):
                print(f"\n       {k}: {json.dumps(v, default=str)[:120]}")
            else:
                print(f"\n       {k}: {str(v)[:120]}")


# ---------------------------------------------------------------------------
# Optional: run fairness check before the main pipeline
# ---------------------------------------------------------------------------

def run_fairness_checks() -> None:
    print_banner("GUARDRAIL #4 · FAIRNESS CHECK (name-swap test)")
    for candidate in CANDIDATES:
        fairness_ok(
            resume_text=candidate["resume"],
            rubric=RUBRIC,
            original_name=candidate["name"].split()[0],   # first name only
        )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main(auto_approve: bool = False, run_fairness: bool = False) -> None:
    run_id = str(uuid.uuid4())[:8]
    print_banner(f"TechVest Recruitment Agent  |  run={run_id}")

    # --- Optional fairness check (guardrail #4) ---
    if run_fairness:
        run_fairness_checks()

    # --- Build graph ---
    app = build_graph()

    # --- Inputs ---
    inputs: dict = {
        "jd": JD,
        "rubric": RUBRIC,
        "candidates": CANDIDATES,
        "parsed_profiles": [],
        "scorecards": [],
        "shortlist": [],
        "trajectory": [],
        "injection_flags": [],
    }

    # Guardrail #2 — recursion limit is the hard stop
    cfg = {
        "configurable": {"thread_id": run_id},
        "recursion_limit": 15,
    }

    # --- Phase 1: run until human gate (parse → score → decide) ---
    print_banner("PHASE 1 · parse → score → decide")
    app.invoke(inputs, config=cfg)

    # Inspect state at the pause point
    snap = app.get_state(cfg)
    assert snap.next == ("schedule",), (
        f"Expected graph to pause at 'schedule', but next={snap.next}. "
        "Check graph wiring."
    )

    state_at_gate = snap.values
    print("\n  Graph paused at 'schedule' node (human gate active).")
    print(f"  Injection flags: {state_at_gate.get('injection_flags', [])}")
    print_shortlist(state_at_gate["shortlist"])
    print_trajectory(state_at_gate["trajectory"])

    # --- Guardrail #1: human approval ---
    if auto_approve:
        print("\n  [--auto-approve] Proceeding without human prompt.")
        approved = True
    else:
        print("\n" + "-" * 70)
        answer = input(
            "  HUMAN GATE: Review the shortlist above.\n"
            "  Approve interview proposals? [y/N]: "
        ).strip().lower()
        approved = answer == "y"

    if not approved:
        print("\n  Human rejected or skipped. propose_interview will NOT fire.")
        # Persist pre-schedule audit
        persist_audit(state_at_gate, run_id + "_pre_gate")
        print_banner("Run complete (no interviews proposed — human declined)")
        return

    # --- Phase 2: resume graph → schedule node fires ---
    print_banner("PHASE 2 · schedule (propose_interview — human approved)")
    app.invoke(None, config=cfg)   # None = resume from checkpoint

    final_snap = app.get_state(cfg)
    final_state = final_snap.values

    print_shortlist(final_state["shortlist"])
    print_trajectory(final_state["trajectory"])

    # --- Guardrail #5: persist full audit log ---
    persist_audit(final_state, run_id)

    print_banner(f"Run complete  |  run_id={run_id}")
    print(f"  Audit log: audit/run_{run_id}.json\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TechVest Recruitment Agent")
    parser.add_argument(
        "--auto-approve",
        action="store_true",
        help="Skip human gate prompt (useful for CI / demo)",
    )
    parser.add_argument(
        "--fairness-check",
        action="store_true",
        help="Run name-swap fairness test before main pipeline",
    )
    args = parser.parse_args()

    main(auto_approve=args.auto_approve, run_fairness=args.fairness_check)
