"""
Day 7 · Exercise 3–5 — Multi-Agent Crew Graph
===============================================
Wires all agents into a LangGraph with:

  coordinator -> analyst -> scorer
                              |
                  (borderline?) -> verifier <-> scorer/analyst (feedback loop, max 3 revisions)
                              |
                            decider -> [human gate] -> scheduler -> END
                              |
                  (no interviews) -> END

Guardrails:
  #1 Human gate      — interrupt_before=['scheduler']
  #2 Step budget     — recursion_limit=25 (multi-agent; ~4 agents x 3 candidates + retries)
  #3 Injection       — in analyst_node (flagged, not executed)
  #4 Fairness        — in verifier_node (name-swap re-score)
  #5 Audit log       — persist_audit() in main runner / app
"""

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from agents import (
    analyst_node,
    coordinator_node,
    decider_node,
    human_escalation_node,
    route_after_decider,
    route_after_scorer,
    route_after_verifier,
    scheduler_node,
    scorer_node,
    verifier_node,
)
from crew_state import CrewState

# ── Step budget (Exercise 5) ──────────────────────────────────────────────────
# ~4 agents × 3 candidates + up to 3 verifier retries + coordinator + scheduler
STEP_BUDGET = 25


def build_crew(interrupt_before_scheduler: bool = True):
    """
    Build and compile the multi-agent recruitment crew graph.

    Args:
        interrupt_before_scheduler: If True, graph pauses for human approval
            before the scheduler fires. Set False only for automated tests.
    """
    g = StateGraph(CrewState)

    # Register all nodes
    g.add_node("coordinator",       coordinator_node)
    g.add_node("analyst",           analyst_node)
    g.add_node("scorer",            scorer_node)
    g.add_node("verifier",          verifier_node)
    g.add_node("decider",           decider_node)
    g.add_node("scheduler",         scheduler_node)
    g.add_node("human_escalation",  human_escalation_node)

    # ── Entry point ────────────────────────────────────────────────────────
    g.set_entry_point("coordinator")

    # ── Linear edges ──────────────────────────────────────────────────────
    g.add_edge("coordinator", "analyst")
    g.add_edge("analyst",     "scorer")

    # ── Conditional: after scorer → verifier | decider | analyst (retry) ──
    g.add_conditional_edges(
        "scorer",
        route_after_scorer,
        {
            "verifier":          "verifier",
            "decider":           "decider",
            "analyst":           "analyst",        # retry: bad handoff
            "human_escalation":  "human_escalation",
        },
    )

    # ── Conditional: after verifier → decider | analyst | scorer (send-back) ─
    g.add_conditional_edges(
        "verifier",
        route_after_verifier,
        {
            "decider":           "decider",
            "analyst":           "analyst",        # bad profile
            "scorer":            "scorer",         # bad score
            "human_escalation":  "human_escalation",
        },
    )

    # ── Conditional: after decider → scheduler | END ──────────────────────
    g.add_conditional_edges(
        "decider",
        route_after_decider,
        {
            "scheduler": "scheduler",
            "END":        END,
        },
    )

    # ── Terminal edges ──────────────────────────────────────────────────────
    g.add_edge("scheduler",         END)
    g.add_edge("human_escalation",  END)

    # ── Compile ────────────────────────────────────────────────────────────
    interrupt_nodes = ["scheduler"] if interrupt_before_scheduler else []
    app = g.compile(
        checkpointer=MemorySaver(),
        interrupt_before=interrupt_nodes,
    )
    return app


def make_inputs(jd: dict, rubric: dict, candidates: list[dict]) -> dict:
    """Return a clean initial state dict for crew.invoke()."""
    return {
        "jd":               jd,
        "rubric":           rubric,
        "candidates":       candidates,
        "parsed_profiles":  [],
        "scorecards":       [],
        "verified_scores":  [],
        "revision_count":   0,
        "shortlist":        [],
        "injection_flags":  [],
        "agent_trace":      [],
    }


def make_config(thread_id: str) -> dict:
    """Return invoke config with step budget."""
    return {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": STEP_BUDGET,
    }
