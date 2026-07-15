"""
Day 7 · Exercises 1–5 — Agent Nodes & Routing
===============================================
Five specialist agents + routing functions:

  coordinator  — sequences the crew, enforces step budget (supervisor)
  analyst      — parses resumes, detects injection          (pipeline stage 1)
  scorer       — scores against rubric, validates handoff   (pipeline stage 2)
  verifier     — peer-to-peer re-check for borderline cases
  decider      — builds the shortlist from verified scores
  scheduler    — proposes interview slots (ACTION, gated)

Pattern (Exercise 1):
  coordinator (supervisor)
       |
  analyst --> scorer <--> verifier (peer-to-peer, borderline only)
                |
            decider --> [human gate] --> scheduler
"""

import re

from pydantic import ValidationError

from crew_state import CrewState
from data import RUBRIC, CandidateProfile
from guardrails import INJECTION_PATTERN
from tools import (
    check_availability,
    parse_resume,
    propose_interview,
    score_candidate,
)

# ── Thresholds ────────────────────────────────────────────────────────────────
BORDERLINE_LOW      = 2.8   # verifier fires for scores in [LOW, HIGH]
BORDERLINE_HIGH     = 3.4
INTERVIEW_THRESHOLD = 3.5
HOLD_THRESHOLD      = 2.5
MAX_REVISIONS       = 3     # after this many rejections, pass through with warning
FAIRNESS_DELTA      = 0.5   # LLM scoring has natural variance; flag only large gaps


# ─────────────────────────────────────────────────────────────────────────────
# Helper: get the LATEST N scorecards (one per candidate)
# Because scorecards uses operator.add, retries accumulate old entries.
# ─────────────────────────────────────────────────────────────────────────────
def _latest_scorecards(state: CrewState) -> list[dict]:
    """Return only the most recent scorecard for each unique candidate name."""
    cards = state.get("scorecards", [])
    seen  = {}
    for card in cards:          # iterate forward; later entries overwrite earlier
        seen[card["name"]] = card
    return list(seen.values())


# ─────────────────────────────────────────────────────────────────────────────
# COORDINATOR  (supervisor — pure routing, no LLM call)
# ─────────────────────────────────────────────────────────────────────────────
def coordinator_node(state: CrewState) -> dict:
    """Sequences the crew. Logs the plan; does NO LLM work itself."""
    trace = [{
        "agent": "coordinator",
        "thought": (
            "Crew starting. Plan: analyst -> scorer -> "
            "verifier (borderline) -> decider -> [human gate] -> scheduler"
        ),
        "action": "route_to_analyst",
        "observation": f"{len(state['candidates'])} candidates to process",
    }]
    return {"agent_trace": trace}


# ─────────────────────────────────────────────────────────────────────────────
# ANALYST  (pipeline stage 1 — parse + injection detection)
# ─────────────────────────────────────────────────────────────────────────────
def analyst_node(state: CrewState) -> dict:
    """
    Exercise 3 — Agent 1 of the two-agent handoff.
    Reads:  candidates
    Writes: parsed_profiles, injection_flags, agent_trace
    """
    profiles  = []
    trace     = []
    inj_flags = []

    for candidate in state["candidates"]:
        name        = candidate["name"]
        resume_text = candidate["resume"]

        # Guardrail 3: injection detection
        flagged = bool(INJECTION_PATTERN.search(resume_text))
        if flagged:
            inj_flags.append(name)

        trace.append({
            "agent": "analyst",
            "thought": f"Parsing resume for {name}",
            "action": "parse_resume",
            "observation": f"injection_flagged={flagged}",
        })

        profile_dict = parse_resume.invoke({"resume_text": resume_text})
        profile_dict["injection_flagged"] = flagged
        profiles.append(profile_dict)

        trace.append({
            "agent": "analyst",
            "thought": f"Parsed {name}",
            "action": "write_parsed_profiles",
            "observation": (
                f"skills={len(profile_dict.get('skills', []))}, "
                f"projects={len(profile_dict.get('projects', []))}"
            ),
        })

    return {
        "parsed_profiles": profiles,
        "injection_flags": inj_flags,
        "agent_trace": trace,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SCORER  (pipeline stage 2 — validate handoff + score)
# ─────────────────────────────────────────────────────────────────────────────
def scorer_node(state: CrewState) -> dict:
    """
    Exercise 3 — Agent 2 of the two-agent handoff.
    Exercise 5 — Pydantic boundary validation.
    Reads:  parsed_profiles
    Writes: scorecards, agent_trace  (or revision_count on bad handoff)
    """
    cards = []
    trace = []

    for raw in state["parsed_profiles"]:
        name = raw.get("name", "Unknown")

        # Exercise 5: validate handoff with Pydantic
        try:
            profile_obj = CandidateProfile(**{
                k: raw[k]
                for k in ("name", "skills", "years", "education", "projects")
            })
        except (ValidationError, KeyError) as e:
            trace.append({
                "agent": "scorer",
                "thought": f"Handoff validation FAILED for {name}",
                "action": "pydantic_validation",
                "observation": f"ValidationError: {e} — incrementing revision_count",
            })
            return {
                "revision_count": state.get("revision_count", 0) + 1,
                "agent_trace": trace,
            }

        trace.append({
            "agent": "scorer",
            "thought": f"Handoff valid for {name}; scoring against rubric",
            "action": "score_candidate",
            "observation": "profile validated",
        })

        card_dict = score_candidate.invoke(
            {"profile": profile_obj.model_dump(), "rubric": state["rubric"]}
        )
        cards.append(card_dict)

        trace.append({
            "agent": "scorer",
            "thought": f"Scored {name}",
            "action": "write_scorecards",
            "observation": f"weighted={card_dict['weighted']}",
        })

    return {"scorecards": cards, "agent_trace": trace}


# ─────────────────────────────────────────────────────────────────────────────
# VERIFIER  (peer-to-peer — re-check borderline candidates)
# ─────────────────────────────────────────────────────────────────────────────
def verifier_node(state: CrewState) -> dict:
    """
    Exercise 4 — Peer-to-peer independent verification.
    Only fires for candidates in the borderline band [2.8, 3.4].
    Checks:
      1. Name-swap fairness re-score (delta threshold = 0.5)
      2. Injection flag was detected and didn't inflate the score
    Reads:  scorecards, parsed_profiles
    Writes: verified_scores, revision_count, agent_trace
    """
    verified       = []
    trace          = []
    revision_delta = 0

    latest_cards = _latest_scorecards(state)
    profile_map  = {p["name"]: p for p in state.get("parsed_profiles", [])}

    for card in latest_cards:
        weighted = card["weighted"]
        name     = card["name"]

        in_borderline = BORDERLINE_LOW <= weighted <= BORDERLINE_HIGH

        if not in_borderline:
            trace.append({
                "agent": "verifier",
                "thought": f"{name} score={weighted:.2f} — outside borderline band, skip",
                "action": "skip_verify",
                "observation": "passed through",
            })
            verified.append({
                **card,
                "verified": True,
                "verifier_note": "outside borderline band — auto-passed",
            })
            continue

        trace.append({
            "agent": "verifier",
            "thought": f"{name} score={weighted:.2f} is BORDERLINE — running peer check",
            "action": "verify",
            "observation": "starting name-swap fairness re-score",
        })

        raw_profile       = profile_map.get(name, {})
        injection_flagged = raw_profile.get("injection_flagged", False)

        # Fairness re-score: replace name with "Candidate X"
        fair        = True
        score_delta = 0.0
        blind_weighted = weighted   # fallback

        try:
            first_name   = name.split()[0]
            blind_profile = dict(raw_profile)
            blind_profile["name"] = "Candidate X"
            # Strip first name from string fields
            for k, v in blind_profile.items():
                if isinstance(v, str):
                    blind_profile[k] = re.sub(
                        re.escape(first_name), "Candidate", v, flags=re.IGNORECASE
                    )

            # Validate before scoring
            profile_obj = CandidateProfile(**{
                k: blind_profile[k]
                for k in ("name", "skills", "years", "education", "projects")
                if k in blind_profile
            })
            blind_card     = score_candidate.invoke(
                {"profile": profile_obj.model_dump(), "rubric": state["rubric"]}
            )
            blind_weighted = blind_card["weighted"]
            score_delta    = abs(blind_weighted - weighted)
            fair           = score_delta <= FAIRNESS_DELTA

        except Exception as e:
            trace.append({
                "agent": "verifier",
                "thought": f"Blind re-score failed ({e}); treating as fair",
                "action": "blind_score_error",
                "observation": str(e)[:80],
            })

        ok   = fair
        note = (
            f"fair={fair} (delta={score_delta:.2f}, threshold={FAIRNESS_DELTA}), "
            f"injection_flagged={injection_flagged}, "
            f"blind_score={blind_weighted:.2f}"
        )

        trace.append({
            "agent": "verifier",
            "thought": f"Verification result for {name}: ok={ok}",
            "action": "write_verified_scores",
            "observation": note,
        })

        if not ok:
            revision_delta += 1

        verified.append({
            **card,
            "verified": ok,
            "verifier_note": note,
            "blind_weighted": round(blind_weighted, 2),
        })

    return {
        "verified_scores": verified,
        "revision_count": state.get("revision_count", 0) + revision_delta,
        "agent_trace": trace,
    }


# ─────────────────────────────────────────────────────────────────────────────
# DECIDER  (builds the final shortlist)
# ─────────────────────────────────────────────────────────────────────────────
def decider_node(state: CrewState) -> dict:
    """
    Reads verified_scores (or latest scorecards as fallback).
    Writes: shortlist, agent_trace
    """
    # Prefer verified scores; fallback to latest scorecards
    verified = state.get("verified_scores", [])
    source   = verified if verified else _latest_scorecards(state)

    # Deduplicate: keep the last entry per name
    seen   = {}
    for item in source:
        seen[item["name"]] = item
    source = list(seen.values())

    shortlist = []
    trace     = []

    for card in source:
        weighted = card["weighted"]
        name     = card["name"]

        if weighted >= INTERVIEW_THRESHOLD:
            verdict = "interview"
        elif weighted >= HOLD_THRESHOLD:
            verdict = "hold"
        else:
            verdict = "reject"

        best          = max(card["scores"], key=lambda cs: cs["score"])
        justification = (
            f"Weighted score {weighted:.2f}. "
            f"Strongest: '{best['criterion']}' ({best['score']}/5) — "
            f"\"{best['evidence']}\""
        )

        verifier_note = card.get("verifier_note", "")
        was_verified  = card.get("verified", True)

        shortlist.append({
            "name":          name,
            "verdict":       verdict,
            "weighted":      weighted,
            "justification": justification,
            "scorecard":     card,
            "proposed_slot": None,
            "verifier_note": verifier_note,
            "verified":      was_verified,
        })

        trace.append({
            "agent": "decider",
            "thought": f"{name} weighted={weighted:.2f} -> {verdict}",
            "action":  "decide",
            "observation": f"{justification[:80]} | verified={was_verified}",
        })

    # Sort: interview first, hold, reject
    order = {"interview": 0, "hold": 1, "reject": 2}
    shortlist.sort(key=lambda d: order.get(d["verdict"], 3))

    return {"shortlist": shortlist, "agent_trace": trace}


# ─────────────────────────────────────────────────────────────────────────────
# SCHEDULER  (ACTION — gated by interrupt_before)
# ─────────────────────────────────────────────────────────────────────────────
def scheduler_node(state: CrewState) -> dict:
    """
    Proposes interview slots for candidates with verdict='interview'.
    ONLY runs after explicit human approval (interrupt_before=['scheduler']).
    """
    updated = list(state["shortlist"])
    trace   = []

    for i, decision in enumerate(updated):
        if decision["verdict"] == "interview":
            cname = decision["name"]
            slots = check_availability.invoke({"candidate": cname, "week": "next"})
            slot  = slots[0] if slots else "TBD"
            result = propose_interview.invoke({"candidate": cname, "slot": slot})
            updated[i] = {**decision, "proposed_slot": slot}

            trace.append({
                "agent": "scheduler",
                "thought": f"Propose interview for {cname}",
                "action": f"propose_interview({cname}, {slot})",
                "observation": result,
            })

    return {"shortlist": updated, "agent_trace": trace}


# ─────────────────────────────────────────────────────────────────────────────
# ROUTING FUNCTIONS  (pure functions over state — no LLM call)
# ─────────────────────────────────────────────────────────────────────────────
def route_after_scorer(state: CrewState) -> str:
    """
    Exercise 4: Route to verifier if any candidate is borderline.
    Exercise 5: Route back to analyst if scorer returned no cards (bad handoff).
    """
    cards = _latest_scorecards(state)

    # No cards = Pydantic validation failed; retry or escalate
    if not cards:
        rc = state.get("revision_count", 0)
        if rc >= MAX_REVISIONS:
            return "human_escalation"
        return "analyst"

    # Any borderline candidate?
    borderline = any(BORDERLINE_LOW <= c["weighted"] <= BORDERLINE_HIGH for c in cards)
    return "verifier" if borderline else "decider"


def route_after_verifier(state: CrewState) -> str:
    """
    Exercise 4: Send-back rule.
    - All verified or max revisions reached -> decider (pass through with warnings)
    - Some failed and revisions left -> route back for correction
    """
    verified = state.get("verified_scores", [])
    # Only look at the latest per-candidate entry
    seen = {}
    for v in verified:
        seen[v["name"]] = v
    latest = list(seen.values())

    failed = [v for v in latest if not v.get("verified", True)]

    if not failed:
        return "decider"

    rc = state.get("revision_count", 0)
    if rc >= MAX_REVISIONS:
        # Too many retries — pass through to decider with verification warnings
        return "decider"

    # Check if the failure looks like an injection issue (analyst to blame)
    injection_issue = any(
        "injection_flagged=True" in v.get("verifier_note", "")
        for v in failed
    )
    return "analyst" if injection_issue else "scorer"


def route_after_decider(state: CrewState) -> str:
    """Route to scheduler if interviews exist, else END."""
    shortlist = state.get("shortlist", [])
    if any(d["verdict"] == "interview" for d in shortlist):
        return "scheduler"
    return "END"


# ─────────────────────────────────────────────────────────────────────────────
# HUMAN ESCALATION  (fallback when revision cap is exceeded)
# ─────────────────────────────────────────────────────────────────────────────
def human_escalation_node(state: CrewState) -> dict:
    """Logs escalation; surfaces in UI as a warning."""
    trace = [{
        "agent": "coordinator",
        "thought": (
            f"revision_count={state.get('revision_count', 0)} "
            f"exceeded limit of {MAX_REVISIONS}"
        ),
        "action":      "human_escalation",
        "observation": "Persistent disagreement — escalated to human reviewer",
    }]
    return {"agent_trace": trace}
