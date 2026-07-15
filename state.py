"""
Phase 1 · AgentState TypedDict
================================
The single typed state dict threaded through every node in the LangGraph.
The trajectory field is append-only (operator.add) — every node adds its
thought → action → observation entries; nothing is ever overwritten.
"""

import operator
from typing import Annotated, TypedDict

from data import CandidateProfile, Decision, ScoreCard


class AgentState(TypedDict):
    # --- inputs ---
    jd: dict                              # job description
    rubric: dict                          # criterion → weight

    # --- raw data ---
    candidates: list[dict]                # list of {"name": str, "resume": str}

    # --- parsed / scored ---
    parsed_profiles: list[CandidateProfile]
    scorecards: list[ScoreCard]

    # --- decisions ---
    shortlist: list[dict]                 # list of Decision-like dicts (JSON-serialisable)

    # --- append-only audit trail ---
    trajectory: Annotated[list, operator.add]   # every node appends; never overwrites

    # --- guardrail flags ---
    injection_flags: Annotated[list, operator.add]  # names of flagged candidates
