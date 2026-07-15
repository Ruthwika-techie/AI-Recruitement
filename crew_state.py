"""
Day 7 · Exercise 2 — Shared State (Handoff Contract)
======================================================
CrewState is the single typed object every agent reads from and writes to.
Nothing crosses between agents except through this state.

Write / Read map
----------------
Field              Written by          Read by
-----------------  ------------------  ----------------------------
jd                 seeded at start     analyst, scorer
rubric             seeded at start     scorer, verifier
candidates         seeded at start     coordinator, analyst
parsed_profiles    analyst             scorer, verifier
scorecards         scorer              verifier, decider
verified_scores    verifier            decider
revision_count     verifier / router   router (bounds the loop)
shortlist          decider             scheduler, human gate
injection_flags    analyst             verifier, display
agent_trace        every agent         display / audit
"""

import operator
from typing import Annotated, TypedDict


class CrewState(TypedDict):
    # ── Inputs (seeded once, never overwritten) ──────────────────────────
    jd: dict                                   # job description
    rubric: dict                               # criterion -> weight

    # ── Raw data ─────────────────────────────────────────────────────────
    candidates: list[dict]                     # [{"name":str, "resume":str}, ...]

    # ── Analyst output ───────────────────────────────────────────────────
    parsed_profiles: list[dict]                # written by analyst

    # ── Scorer output (reducer: append, not overwrite for parallel runs) ─
    scorecards: Annotated[list[dict], operator.add]

    # ── Verifier output ───────────────────────────────────────────────────
    verified_scores: list[dict]                # written by verifier

    # ── Loop control ──────────────────────────────────────────────────────
    revision_count: int                        # incremented on each rejection

    # ── Final decisions ────────────────────────────────────────────────────
    shortlist: list[dict]                      # written by decider

    # ── Guardrail flags ────────────────────────────────────────────────────
    injection_flags: Annotated[list, operator.add]   # names of flagged candidates

    # ── Observability: append-only trace across ALL agents ────────────────
    agent_trace: Annotated[list, operator.add]       # {agent, thought, action, observation}
