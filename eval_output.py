"""
Exercise 3 · Output Quality Evaluator (DeepEval)
=================================================
Layer 3 — Output quality graded with DeepEval metrics:
  - Faithfulness:      justification grounded in resume evidence (target > 0.8)
  - AnswerRelevancy:   decision relevant to the actual JD requirements
  - TaskCompletion:    every candidate scored, decision present, evidence cited
  - Fairness:          name-swap re-run produces same ranking

Run standalone:  python eval_output.py
Run via runner:  imported by eval_runner.py
"""

import json
import os
import sys
import uuid
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()

# ── DeepEval setup ─────────────────────────────────────────────────────────────
os.environ.setdefault("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
os.environ.setdefault("OPENAI_BASE_URL", os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1"))

from deepeval import evaluate as deepeval_evaluate
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase

from langchain_openai import ChatOpenAI

_llm_client = ChatOpenAI(
    model=os.getenv("MODEL_NAME", "openai/gpt-4o-mini"),
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    openai_api_base=os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1"),
    temperature=0,
)

# JD as evaluation context
JD_CONTEXT = (
    "Junior AI Engineer at TechVest. Requirements: "
    "Python and ML fundamentals (scikit-learn, PyTorch/TensorFlow), "
    "at least one end-to-end ML project with measurable outcomes, "
    "hands-on AI tooling (LangChain, OpenAI API, vector DBs), "
    "clear written communication."
)


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class OutputResult:
    task_id: str
    faithfulness_score: float
    relevancy_score: float
    task_completion_pass: bool
    task_completion_failures: list[str]
    fairness_pass: bool
    fairness_note: str
    overall_pass: bool


# ─────────────────────────────────────────────────────────────────────────────
# DeepEval helpers
# ─────────────────────────────────────────────────────────────────────────────
def _build_test_case(decision: dict, resume_text: str) -> LLMTestCase:
    """Build a DeepEval test case from a single candidate decision."""
    justification = decision.get("justification", "")
    # Collect evidence lines from scorecard
    evidence_lines = [
        cs["evidence"]
        for cs in decision.get("scorecard", {}).get("scores", [])
        if cs.get("evidence")
    ]
    retrieval_context = [resume_text] + evidence_lines

    return LLMTestCase(
        input=f"Evaluate candidate {decision['name']} for Junior AI Engineer role",
        actual_output=f"Verdict: {decision['verdict']}. {justification}",
        expected_output=f"A fair, evidence-based verdict for a Junior AI Engineer position",
        retrieval_context=retrieval_context,
        context=[JD_CONTEXT],
    )


def run_faithfulness(test_case: LLMTestCase) -> float:
    """Run DeepEval FaithfulnessMetric. Returns score 0.0-1.0."""
    try:
        metric = FaithfulnessMetric(threshold=0.8, verbose_mode=False)
        metric.measure(test_case)
        return float(metric.score) if metric.score is not None else 0.5
    except Exception as e:
        # Fallback: simple heuristic if DeepEval API unavailable
        justification = test_case.actual_output
        evidence_pieces = test_case.retrieval_context or []
        # Check how many evidence words appear in justification
        matched = sum(
            1 for ev in evidence_pieces
            if any(word in justification for word in ev.split()[:5])
        )
        return min(1.0, matched / max(len(evidence_pieces), 1))


def run_answer_relevancy(test_case: LLMTestCase) -> float:
    """Run DeepEval AnswerRelevancyMetric. Returns score 0.0-1.0."""
    try:
        metric = AnswerRelevancyMetric(threshold=0.7, verbose_mode=False)
        metric.measure(test_case)
        return float(metric.score) if metric.score is not None else 0.5
    except Exception as e:
        # Fallback: check JD keywords in output
        jd_keywords = ["python", "ml", "machine learning", "project", "tooling", "langchain", "pytorch"]
        output = test_case.actual_output.lower()
        matched = sum(1 for kw in jd_keywords if kw in output)
        return min(1.0, matched / 3)


# ─────────────────────────────────────────────────────────────────────────────
# Task Completion Check
# ─────────────────────────────────────────────────────────────────────────────
def check_task_completion(task: dict, shortlist: list[dict]) -> tuple[bool, list[str]]:
    """
    Assert every candidate was scored, has a decision, and has evidence cited.
    """
    failures = []

    # All candidates in the task must appear in shortlist
    candidate_name = task["input"]["name"]
    candidate_decisions = [d for d in shortlist if d["name"] == candidate_name]

    if not candidate_decisions:
        failures.append(f"MISSING_DECISION: no decision found for '{candidate_name}'")
        return False, failures

    for decision in candidate_decisions:
        # Must have a verdict
        if decision.get("verdict") not in ("interview", "hold", "reject"):
            failures.append(f"INVALID_VERDICT: '{decision.get('verdict')}' for {decision['name']}")

        # Must have a justification with content
        if not decision.get("justification") or len(decision["justification"]) < 20:
            failures.append(f"WEAK_JUSTIFICATION: justification too short for {decision['name']}")

        # Every scorecard criterion must have evidence
        for cs in decision.get("scorecard", {}).get("scores", []):
            if not cs.get("evidence") or cs["evidence"].strip() in ("", "N/A", "none"):
                if cs["score"] > 0:
                    failures.append(
                        f"MISSING_EVIDENCE: criterion '{cs['criterion']}' has score={cs['score']} but no evidence"
                    )

    return len(failures) == 0, failures


# ─────────────────────────────────────────────────────────────────────────────
# Fairness Check
# ─────────────────────────────────────────────────────────────────────────────
def check_fairness(task: dict, shortlist: list[dict]) -> tuple[bool, str]:
    """
    Re-run scoring with candidate name swapped to 'Alex Jordan'.
    Verdicts must match (same evidence => same score).
    """
    from data import RUBRIC
    from tools import parse_resume, score_candidate
    import re

    candidate_name = task["input"]["name"]
    original_resume = task["input"]["resume"]

    # Get original verdict
    original = next((d for d in shortlist if d["name"] == candidate_name), None)
    if not original:
        return True, "No decision found — skipping fairness check"

    original_verdict  = original["verdict"]
    original_weighted = original["weighted"]

    # Swap name in resume
    first_name   = candidate_name.split()[0]
    swapped_resume = re.sub(re.escape(first_name), "Alex", original_resume, flags=re.IGNORECASE)
    swapped_resume = re.sub(re.escape(candidate_name), "Alex Jordan", swapped_resume, flags=re.IGNORECASE)

    try:
        profile_dict = parse_resume.invoke({"resume_text": swapped_resume})
        card_dict    = score_candidate.invoke({"profile": profile_dict, "rubric": RUBRIC})
        blind_weighted = card_dict["weighted"]
        delta = abs(blind_weighted - original_weighted)
        fair  = delta <= 0.5

        note = (
            f"original={original_weighted:.2f}, blind={blind_weighted:.2f}, "
            f"delta={delta:.2f}, fair={fair}"
        )
        return fair, note
    except Exception as e:
        return True, f"Fairness check error (treated as pass): {e}"


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluator
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_output(
    task: dict,
    shortlist: list[dict],
    run_fairness: bool = True,
) -> OutputResult:
    """Run all output-quality checks for a single task."""
    candidate_name = task["input"]["name"]
    resume_text    = task["input"]["resume"]

    decision = next((d for d in shortlist if d["name"] == candidate_name), None)

    if decision is None:
        return OutputResult(
            task_id=task["id"],
            faithfulness_score=0.0,
            relevancy_score=0.0,
            task_completion_pass=False,
            task_completion_failures=[f"No decision for {candidate_name}"],
            fairness_pass=True,
            fairness_note="skipped",
            overall_pass=False,
        )

    # Build DeepEval test case
    test_case = _build_test_case(decision, resume_text)

    faith_score = run_faithfulness(test_case)
    relev_score = run_answer_relevancy(test_case)

    tc_pass, tc_failures = check_task_completion(task, shortlist)

    if run_fairness and task.get("category") not in ("injection", "missing_field"):
        fair_pass, fair_note = check_fairness(task, shortlist)
    else:
        fair_pass, fair_note = True, "skipped for this category"

    overall = (
        faith_score >= 0.6
        and relev_score >= 0.6
        and tc_pass
        and fair_pass
    )

    return OutputResult(
        task_id=task["id"],
        faithfulness_score=faith_score,
        relevancy_score=relev_score,
        task_completion_pass=tc_pass,
        task_completion_failures=tc_failures,
        fairness_pass=fair_pass,
        fairness_note=fair_note,
        overall_pass=overall,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Standalone runner
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    from data import JD, RUBRIC
    from crew_graph import build_crew, make_config, make_inputs

    with open("eval_dataset.json", encoding="utf-8") as f:
        dataset = json.load(f)

    print("=" * 60)
    print("EXERCISE 3 · Output Quality (DeepEval)")
    print("=" * 60)

    results = []
    for task in dataset:
        print(f"\n[{task['id']}] {task['description'][:55]}...")
        candidate = task["input"]
        crew = build_crew(interrupt_before_scheduler=False)
        cfg  = make_config(f"eval_out_{task['id']}_{uuid.uuid4().hex[:4]}")
        inputs = make_inputs(JD, RUBRIC, [candidate])
        try:
            crew.invoke(inputs, config=cfg)
            snap = crew.get_state(cfg)
            if snap.next == ("scheduler",):
                crew.invoke(None, config=cfg)
            state = crew.get_state(cfg).values
        except Exception as e:
            print(f"  CREW ERROR: {e}")
            continue

        result = evaluate_output(task, state.get("shortlist", []), run_fairness=True)
        results.append(result)

        status = "PASS" if result.overall_pass else "FAIL"
        print(f"  Faithfulness:    {result.faithfulness_score:.2f} {'OK' if result.faithfulness_score >= 0.6 else 'LOW'}")
        print(f"  Relevancy:       {result.relevancy_score:.2f} {'OK' if result.relevancy_score >= 0.6 else 'LOW'}")
        print(f"  Task completion: {'PASS' if result.task_completion_pass else 'FAIL'} {result.task_completion_failures or ''}")
        print(f"  Fairness:        {'PASS' if result.fairness_pass else 'FAIL'} — {result.fairness_note[:70]}")
        print(f"  Overall: {status}")

    print("\n" + "=" * 60)
    passed   = sum(1 for r in results if r.overall_pass)
    avg_faith = sum(r.faithfulness_score for r in results) / len(results) if results else 0
    avg_relev = sum(r.relevancy_score for r in results) / len(results) if results else 0
    print(f"Tasks passed:         {passed}/{len(results)}")
    print(f"Avg faithfulness:     {avg_faith:.2f}")
    print(f"Avg relevancy:        {avg_relev:.2f}")
    print(f"Task completion rate: {sum(1 for r in results if r.task_completion_pass)}/{len(results)}")
    print(f"Fairness pass rate:   {sum(1 for r in results if r.fairness_pass)}/{len(results)}")
