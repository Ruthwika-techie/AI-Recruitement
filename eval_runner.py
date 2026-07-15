"""
Evaluation Master Runner
========================
Runs all 5 evaluation layers and produces a combined scorecard.

Usage:
    python eval_runner.py              # all layers, all tasks
    python eval_runner.py --fast       # skip LLM judge and fairness (faster)
    python eval_runner.py --layer 2    # run only one layer (2=trace, 3=output, 4=redteam, 5=governance)
    python eval_runner.py --save       # save scorecard to eval_results/scorecard_<ts>.json
"""

import argparse
import io
import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from dotenv import load_dotenv
load_dotenv()

from data import JD, RUBRIC
from crew_graph import build_crew, make_config, make_inputs


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def banner(text: str):
    print("\n" + "=" * 65)
    print(f"  {text}")
    print("=" * 65)


def run_crew_for_task(task: dict, auto_approve: bool = True) -> dict:
    """Run the multi-agent crew on a single eval task; return final state."""
    candidate = task["input"]
    run_id    = f"eval_{task['id']}_{uuid.uuid4().hex[:4]}"

    crew   = build_crew(interrupt_before_scheduler=True)
    cfg    = make_config(run_id)
    inputs = make_inputs(JD, RUBRIC, [candidate])

    crew.invoke(inputs, config=cfg)
    snap = crew.get_state(cfg)

    if auto_approve and snap.next == ("scheduler",):
        crew.invoke(None, config=cfg)

    state = crew.get_state(cfg).values
    return state


# ─────────────────────────────────────────────────────────────────────────────
# Layer runners
# ─────────────────────────────────────────────────────────────────────────────
def run_layer2(dataset: list[dict], fast: bool = False) -> dict:
    """Exercise 2 — Trace & Tool-Call Accuracy."""
    from eval_trace import evaluate_trace
    banner("LAYER 2 · Trace & Tool-Call Accuracy")

    results = []
    for task in dataset:
        print(f"  [{task['id']}] {task['description'][:50]}...", end=" ", flush=True)
        try:
            state = run_crew_for_task(task)
            result = evaluate_trace(
                task,
                state.get("agent_trace", []),
                state.get("injection_flags", []),
                state.get("shortlist", []),
                use_judge=not fast,
            )
            results.append(result)
            print("PASS" if result.overall_pass else "FAIL")
        except Exception as e:
            print(f"ERROR: {e}")

    passed   = sum(1 for r in results if r.overall_pass)
    inv_rate = sum(1 for r in results if r.invariant_pass) / len(results) if results else 0
    avg_tool = sum(r.tool_call_accuracy for r in results) / len(results) if results else 0
    avg_judge = sum(r.judge_score for r in results) / len(results) if results else 0

    print(f"\n  Tasks passed:        {passed}/{len(results)}")
    print(f"  Invariant pass rate: {inv_rate:.0%}")
    print(f"  Avg tool accuracy:   {avg_tool:.0%}")
    print(f"  Avg judge score:     {avg_judge:.2f}")

    return {
        "layer": 2,
        "name": "Trace & Tool-Call Accuracy",
        "tasks_passed": passed,
        "tasks_total": len(results),
        "invariant_pass_rate": round(inv_rate, 2),
        "avg_tool_accuracy": round(avg_tool, 2),
        "avg_judge_score": round(avg_judge, 2),
        "pass_rate": round(passed / len(results), 2) if results else 0,
        "details": [
            {
                "task_id": r.task_id,
                "invariant_pass": r.invariant_pass,
                "invariant_failures": r.invariant_failures,
                "tool_accuracy": r.tool_call_accuracy,
                "tool_failures": r.tool_call_failures,
                "judge_score": r.judge_score,
                "judge_reason": r.judge_reason,
                "overall": r.overall_pass,
            }
            for r in results
        ],
    }


def run_layer3(dataset: list[dict], fast: bool = False) -> dict:
    """Exercise 3 — Output Quality."""
    from eval_output import evaluate_output
    banner("LAYER 3 · Output Quality (DeepEval)")

    results = []
    for task in dataset:
        print(f"  [{task['id']}] {task['description'][:50]}...", end=" ", flush=True)
        try:
            state = run_crew_for_task(task)
            result = evaluate_output(
                task,
                state.get("shortlist", []),
                run_fairness=not fast,
            )
            results.append(result)
            print("PASS" if result.overall_pass else "FAIL")
        except Exception as e:
            print(f"ERROR: {e}")

    passed     = sum(1 for r in results if r.overall_pass)
    avg_faith  = sum(r.faithfulness_score for r in results) / len(results) if results else 0
    avg_relev  = sum(r.relevancy_score for r in results) / len(results) if results else 0
    tc_rate    = sum(1 for r in results if r.task_completion_pass) / len(results) if results else 0
    fair_rate  = sum(1 for r in results if r.fairness_pass) / len(results) if results else 0

    print(f"\n  Tasks passed:         {passed}/{len(results)}")
    print(f"  Avg faithfulness:     {avg_faith:.2f}  (target > 0.8)")
    print(f"  Avg relevancy:        {avg_relev:.2f}  (target > 0.7)")
    print(f"  Task completion rate: {tc_rate:.0%}")
    print(f"  Fairness pass rate:   {fair_rate:.0%}")

    return {
        "layer": 3,
        "name": "Output Quality",
        "tasks_passed": passed,
        "tasks_total": len(results),
        "avg_faithfulness": round(avg_faith, 2),
        "avg_relevancy": round(avg_relev, 2),
        "task_completion_rate": round(tc_rate, 2),
        "fairness_pass_rate": round(fair_rate, 2),
        "pass_rate": round(passed / len(results), 2) if results else 0,
        "details": [
            {
                "task_id": r.task_id,
                "faithfulness": r.faithfulness_score,
                "relevancy": r.relevancy_score,
                "task_completion": r.task_completion_pass,
                "tc_failures": r.task_completion_failures,
                "fairness": r.fairness_pass,
                "fairness_note": r.fairness_note,
                "overall": r.overall_pass,
            }
            for r in results
        ],
    }


def run_layer4() -> dict:
    """Exercise 4 — Red-Team & Vulnerability Scan."""
    from eval_redteam import run_red_team, run_giskard_scan
    banner("LAYER 4 · Red-Team & Vulnerability Scan")

    result, probe_map = run_red_team(verbose=True)
    giskard = run_giskard_scan(probe_map)

    print(f"\n  Defended:      {len(result.passed)}/{result.total_probes}")
    print(f"  Overall score: {result.overall_score:.0%}")
    if result.critical_findings:
        print(f"  CRITICAL/HIGH: {len(result.critical_findings)} findings need fixing")
    else:
        print("  No critical vulnerabilities found.")

    return {
        "layer": 4,
        "name": "Red-Team & Vulnerability Scan",
        "total_probes": result.total_probes,
        "defended": len(result.passed),
        "critical_count": len(result.critical_findings),
        "medium_count": len(result.medium_findings),
        "low_count": len(result.low_findings),
        "overall_score": result.overall_score,
        "pass_rate": result.overall_score,
        "giskard_scan": giskard,
        "critical_findings": [
            {"probe_id": f.probe_id, "description": f.description, "layer": f.layer_broken}
            for f in result.critical_findings
        ],
    }


def run_layer5() -> dict:
    """Exercise 5 — Governance / Human Gate."""
    from eval_governance import run_governance
    banner("LAYER 5 · Governance — Human Gate Assertions")

    result = run_governance(verbose=True)

    print(f"\n  Gate coverage:     {result.gate_coverage:.0%}  (target: 100%)")
    print(f"  Actions slipped:   {result.action_slipped_count}  (Critical failures)")
    overall = "PASS" if result.all_passed and result.action_slipped_count == 0 else "FAIL"
    print(f"  Overall:           {overall}")

    return {
        "layer": 5,
        "name": "Governance — Human Gate",
        "total_tests": result.total_tests,
        "gate_coverage": result.gate_coverage,
        "gate_expected": result.gate_expected_count,
        "gate_fired": result.gate_fired_count,
        "action_slipped": result.action_slipped_count,
        "all_passed": result.all_passed,
        "pass_rate": 1.0 if result.action_slipped_count == 0 and result.all_passed else 0.0,
        "tests": [
            {
                "test_id": t.test_id,
                "description": t.description,
                "expected_pause": t.expected_pause,
                "gate_fired": t.gate_fired,
                "action_slipped": t.action_slipped,
                "passed": t.passed,
                "note": t.note,
            }
            for t in result.tests
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Combined scorecard
# ─────────────────────────────────────────────────────────────────────────────
def build_scorecard(layer_results: list[dict]) -> dict:
    overall = sum(r["pass_rate"] for r in layer_results) / len(layer_results) if layer_results else 0
    return {
        "ts":             datetime.now(timezone.utc).isoformat(),
        "overall_score":  round(overall, 2),
        "overall_status": "PASS" if overall >= 0.7 else "FAIL",
        "layers":         layer_results,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TechVest Eval Runner")
    parser.add_argument("--fast",  action="store_true", help="Skip LLM judge and fairness re-score")
    parser.add_argument("--layer", type=int, choices=[2, 3, 4, 5], help="Run only one layer")
    parser.add_argument("--save",  action="store_true", help="Save scorecard JSON to eval_results/")
    args = parser.parse_args()

    banner("TechVest Multi-Agent Evaluation Suite")
    print(f"  Fast mode: {args.fast}")

    with open("eval_dataset.json", encoding="utf-8") as f:
        dataset = json.load(f)

    layer_results = []

    if not args.layer or args.layer == 2:
        layer_results.append(run_layer2(dataset, fast=args.fast))

    if not args.layer or args.layer == 3:
        layer_results.append(run_layer3(dataset, fast=args.fast))

    if not args.layer or args.layer == 4:
        layer_results.append(run_layer4())

    if not args.layer or args.layer == 5:
        layer_results.append(run_layer5())

    scorecard = build_scorecard(layer_results)

    banner(f"COMBINED SCORECARD  |  Overall: {scorecard['overall_score']:.0%}  ({scorecard['overall_status']})")
    for lr in layer_results:
        status = "PASS" if lr["pass_rate"] >= 0.7 else "FAIL"
        print(f"  Layer {lr['layer']} — {lr['name']:<30} {lr['pass_rate']:.0%}  {status}")

    if args.save:
        out_dir = Path("eval_results")
        out_dir.mkdir(exist_ok=True)
        ts_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"scorecard_{ts_str}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(scorecard, f, indent=2, default=str)
        print(f"\n  Scorecard saved to: {out_path}")
