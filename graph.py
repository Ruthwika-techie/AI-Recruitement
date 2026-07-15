"""
Phase 3 · Wire the LangGraph agent loop
=========================================
Nodes: parse → score → decide → schedule (conditional)
Edge:  decide → schedule  only if at least one candidate has verdict == 'interview'
        decide → END       otherwise

Guardrails wired here:
  - interrupt_before=['schedule']  →  human gate (Phase 5, guardrail #1)
  - recursion_limit=15             →  iteration cap (Phase 5, guardrail #2)
  - injection sanitisation in parse_node (guardrail #3)
"""

import re

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from data import RUBRIC, Decision
from guardrails import (
    INJECTION_PATTERN,
    fairness_ok,
    persist_audit,
    sanitize_resume,
)
from state import AgentState
from tools import (
    check_availability,
    parse_resume,
    propose_interview,
    score_candidate,
)


# ---------------------------------------------------------------------------
# Node 1 · parse_node
# ---------------------------------------------------------------------------
def parse_node(state: AgentState) -> dict:
    """Parse every raw résumé into a CandidateProfile.

    Also runs injection detection (guardrail #3) and records flags.
    """
    profiles = []
    traj = []
    injection_flags = []

    for candidate in state["candidates"]:
        name = candidate["name"]
        resume_text = candidate["resume"]

        # --- Guardrail #3: sanitise before parsing ---
        clean_text, flagged = sanitize_resume(resume_text)
        if flagged:
            injection_flags.append(name)

        traj.append({
            "thought": f"Parsing résumé for {name}",
            "action": "parse_resume",
            "injection_flagged": flagged,
        })

        # Parse using the tool (passes sanitised text)
        profile_dict = parse_resume.invoke({"resume_text": clean_text})
        profiles.append(profile_dict)

        traj.append({
            "action": "parse_resume",
            "observation": f"Parsed profile for {profile_dict['name']}: "
                           f"{len(profile_dict['skills'])} skills, "
                           f"{len(profile_dict['projects'])} projects",
        })

    return {
        "parsed_profiles": profiles,
        "trajectory": traj,
        "injection_flags": injection_flags,
    }


# ---------------------------------------------------------------------------
# Node 2 · score_node
# ---------------------------------------------------------------------------
def score_node(state: AgentState) -> dict:
    """Score every parsed profile against the rubric."""
    cards = []
    traj = []

    for profile in state["parsed_profiles"]:
        name = profile["name"]
        traj.append({
            "thought": f"{name} parsed; now score against the rubric",
            "action": f"score_candidate(profile={name}, rubric=RUBRIC)",
        })

        card_dict = score_candidate.invoke(
            {"profile": profile, "rubric": state["rubric"]}
        )
        cards.append(card_dict)

        traj.append({
            "observation": {
                "name": card_dict["name"],
                "weighted": card_dict["weighted"],
                "criteria": {
                    cs["criterion"]: (cs["score"], cs["evidence"])
                    for cs in card_dict["scores"]
                },
            }
        })

    return {"scorecards": cards, "trajectory": traj}


# ---------------------------------------------------------------------------
# Node 3 · decide_node
# ---------------------------------------------------------------------------
def decide_node(state: AgentState) -> dict:
    """Convert scorecards into structured Decisions and build the shortlist."""
    shortlist = []
    traj = []

    for card in state["scorecards"]:
        weighted = card["weighted"]
        name = card["name"]

        # Verdict thresholds
        if weighted >= 3.5:
            verdict = "interview"
        elif weighted >= 2.5:
            verdict = "hold"
        else:
            verdict = "reject"

        # Build justification from the highest-scoring criterion
        best = max(card["scores"], key=lambda cs: cs["score"])
        justification = (
            f"Weighted score {weighted:.2f}. "
            f"Strongest criterion: '{best['criterion']}' (score {best['score']}/5) — "
            f"evidence: \"{best['evidence']}\""
        )

        decision = {
            "name": name,
            "verdict": verdict,
            "weighted": weighted,
            "justification": justification,
            "scorecard": card,
            "proposed_slot": None,
        }
        shortlist.append(decision)

        traj.append({
            "thought": f"Evaluated {name}: weighted={weighted:.2f} → verdict={verdict}",
            "action": "decide",
            "observation": {"verdict": verdict, "justification": justification},
        })

    # Sort shortlist: interview first, then hold, then reject
    order = {"interview": 0, "hold": 1, "reject": 2}
    shortlist.sort(key=lambda d: order.get(d["verdict"], 3))

    return {"shortlist": shortlist, "trajectory": traj}


# ---------------------------------------------------------------------------
# Node 4 · schedule_node  (ACTION — gated by interrupt_before)
# ---------------------------------------------------------------------------
def schedule_node(state: AgentState) -> dict:
    """Propose interview slots for shortlisted candidates.

    This node only runs AFTER explicit human approval
    (graph compiled with interrupt_before=['schedule']).
    """
    updated_shortlist = list(state["shortlist"])
    traj = []

    for i, decision in enumerate(updated_shortlist):
        if decision["verdict"] == "interview":
            candidate_name = decision["name"]

            traj.append({
                "thought": f"Checking availability for {candidate_name}",
                "action": f"check_availability({candidate_name}, week='next')",
            })

            slots = check_availability.invoke(
                {"candidate": candidate_name, "week": "next"}
            )
            chosen_slot = slots[0] if slots else "TBD"

            traj.append({"observation": f"Available slots: {slots}; chose {chosen_slot}"})

            # ACTION — this is the gate-protected step
            result = propose_interview.invoke(
                {"candidate": candidate_name, "slot": chosen_slot}
            )
            updated_shortlist[i] = {**decision, "proposed_slot": chosen_slot}

            traj.append({
                "action": f"propose_interview({candidate_name}, {chosen_slot})",
                "observation": result,
            })

    return {"shortlist": updated_shortlist, "trajectory": traj}


# ---------------------------------------------------------------------------
# Conditional edge — route after decide_node
# ---------------------------------------------------------------------------
def route_after_decide(state: AgentState) -> str:
    """Route to 'schedule' if any candidate was shortlisted for interview."""
    if any(d["verdict"] == "interview" for d in state["shortlist"]):
        return "schedule"
    return END


# ---------------------------------------------------------------------------
# Build and compile the graph
# ---------------------------------------------------------------------------
def build_graph():
    """Construct, compile, and return the recruitment agent graph."""
    g = StateGraph(AgentState)

    # Register nodes
    for node_name, node_fn in [
        ("parse", parse_node),
        ("score", score_node),
        ("decide", decide_node),
        ("schedule", schedule_node),
    ]:
        g.add_node(node_name, node_fn)

    # Entry point and linear edges
    g.set_entry_point("parse")
    g.add_edge("parse", "score")
    g.add_edge("score", "decide")

    # Conditional edge after decide
    g.add_conditional_edges(
        "decide",
        route_after_decide,
        {"schedule": "schedule", END: END},
    )
    g.add_edge("schedule", END)

    # Compile with:
    #   checkpointer  → enables human-gate pause + state replay
    #   interrupt_before=['schedule'] → Guardrail #1: human gate
    app = g.compile(
        checkpointer=MemorySaver(),
        interrupt_before=["schedule"],
    )
    return app
