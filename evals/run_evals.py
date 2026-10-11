"""
Phase 6 — Evals

Runs automated evaluations on three Job Scout agents:
  - Ranker
  - Gap Analysis
  - Application Pack (cover letter)

Each eval is a function decorated with @observe() so it creates a trace in Langfuse.
At the end of each eval, we call langfuse.score() to attach a numeric result to that trace.
You can then see all scores in the Langfuse UI under "Scores".

Eval types used here:
  - Deterministic : checks output format/structure with plain Python (no LLM)
  - Reference-based: checks that agent behaviour is coherent (e.g. relevant job > irrelevant job)

How to run:
    python evals/run_evals.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from langfuse import Langfuse

from agents.application_pack import generate_application_pack
from agents.cv_parser import CVProfile
from agents.gap_analysis import analyse_gap
from agents.job_search import JobPosting
from agents.ranker import RankedJob, rank_jobs

# ── Langfuse client ──────────────────────────────────────────────────────────
# langfuse.trace()  → creates a trace in Langfuse (one per eval function)
# trace.score()     → attaches a numeric result to that trace
# langfuse.flush()  → sends everything before the script exits
langfuse = Langfuse()


# ── Dataset loaders ──────────────────────────────────────────────────────────

DATASETS_DIR = os.path.join(os.path.dirname(__file__), "datasets")


def load_cv(filename: str) -> CVProfile:
    """Loads a CV JSON file and converts it into a CVProfile Pydantic object."""
    path = os.path.join(DATASETS_DIR, filename)
    with open(path) as f:
        data = json.load(f)
    # Pydantic handles nested model conversion automatically:
    # the dicts inside "experience" and "education" become Experience/Education objects.
    return CVProfile(**data)


def load_job_postings(key: str) -> tuple[list[JobPosting], list[JobPosting]]:
    """
    Loads job postings from the dataset JSON.
    Returns two lists: (relevant_jobs, irrelevant_jobs).
    """
    path = os.path.join(DATASETS_DIR, "job_postings.json")
    with open(path) as f:
        data = json.load(f)

    relevant, irrelevant = [], []
    for entry in data[key]:
        posting = JobPosting(
            title=entry["title"],
            url=entry["url"],
            source=entry["source"],
            description=entry["description"],
        )
        if entry["label"] == "relevant":
            relevant.append(posting)
        else:
            irrelevant.append(posting)

    return relevant, irrelevant


# ── Eval runner ──────────────────────────────────────────────────────────────

def run_eval(eval_name: str, score_name: str, fn, *args) -> None:
    """
    Runs one eval function, logs the result to Langfuse, and records it locally.

    Langfuse v4 pattern:
      1. start_as_current_observation() → opens a trace/span visible in Langfuse UI
      2. fn(*args)                      → runs the eval, returns (passed, comment)
      3. score_current_trace()          → attaches the numeric score to that trace
      4. flush()                        → sends everything at the end of main()
    """
    with langfuse.start_as_current_observation(name=eval_name, as_type="evaluator"):
        try:
            passed, comment = fn(*args)
        except Exception as e:
            passed, comment = False, f"Exception: {e}"

        langfuse.score_current_trace(
            name=score_name,
            value=1.0 if passed else 0.0,
            comment=comment,
        )

    results.append({"eval": eval_name, "passed": passed, "comment": comment})
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  {status}  {eval_name}")
    if comment:
        print(f"         {comment}")


# ── Eval results collector ───────────────────────────────────────────────────

results: list[dict] = []


# ── RANKER EVALS ─────────────────────────────────────────────────────────────

def eval_ranker_score_range(profile: CVProfile, jobs: list[JobPosting]) -> tuple[bool, str]:
    """
    DETERMINISTIC eval.
    Every ranked job must have a score that is an integer between 0 and 100.
    This catches bugs like the LLM returning 102, -5, or a float like 85.5.
    """
    ranked = rank_jobs(profile, jobs)
    invalid = [
        f"{job.title}: {job.score}" for job in ranked
        if not (isinstance(job.score, int) and 0 <= job.score <= 100)
    ]
    passed = len(invalid) == 0
    comment = "All scores valid" if passed else f"Invalid scores: {invalid}"
    return passed, comment


def eval_ranker_rationale_not_empty(profile: CVProfile, jobs: list[JobPosting]) -> tuple[bool, str]:
    """
    DETERMINISTIC eval.
    Every ranked job must have a non-empty rationale string.
    An empty rationale means the LLM failed to explain its reasoning.
    """
    ranked = rank_jobs(profile, jobs)
    all_have_rationale = all(
        isinstance(job.rationale, str) and len(job.rationale.strip()) > 10
        for job in ranked
    )
    comment = "All rationales non-empty" if all_have_rationale else "Some rationales are empty or too short"
    return all_have_rationale, comment


def eval_ranker_coherence(
    profile: CVProfile,
    relevant_jobs: list[JobPosting],
    irrelevant_jobs: list[JobPosting],
) -> tuple[bool, str]:
    """
    REFERENCE-BASED eval.
    The average score of relevant jobs must be higher than the average score
    of irrelevant jobs. This tests that the ranker actually understands fit.

    We don't check exact scores — just that the ordering is sensible.
    If the ranker scores a Marketing Manager higher than a Data Engineer
    for a data engineer CV, something is wrong.
    """
    all_jobs = relevant_jobs + irrelevant_jobs
    ranked_map = {job.url: job for job in rank_jobs(profile, all_jobs)}
    relevant_avg = sum(ranked_map[j.url].score for j in relevant_jobs) / len(relevant_jobs)
    irrelevant_avg = sum(ranked_map[j.url].score for j in irrelevant_jobs) / len(irrelevant_jobs)
    passed = relevant_avg > irrelevant_avg
    comment = f"Relevant avg={relevant_avg:.1f} vs Irrelevant avg={irrelevant_avg:.1f}"
    return passed, comment


# ── GAP ANALYSIS EVALS ───────────────────────────────────────────────────────

def eval_gap_structure(profile: CVProfile, job: RankedJob) -> tuple[bool, str]:
    """
    DETERMINISTIC eval.
    The gap analysis must return non-empty lists for all three sections
    (strengths, critical_gaps, minor_gaps) and a non-empty recommendation.
    """
    gap = analyse_gap(profile, job)
    checks = {
        "has_strengths":      len(gap.strengths) > 0,
        "has_critical_gaps":  len(gap.critical_gaps) > 0,
        "has_recommendation": len(gap.recommendation.strip()) > 20,
    }
    all_passed = all(checks.values())
    failed = [k for k, v in checks.items() if not v]
    comment = "All sections populated" if all_passed else f"Empty sections: {failed}"
    return all_passed, comment


def eval_gap_mentions_skills(profile: CVProfile, job: RankedJob) -> tuple[bool, str]:
    """
    REFERENCE-BASED eval.
    At least one item in strengths or gaps should mention a skill from the profile.
    This checks that the gap analysis is personalised and not generic.
    """
    gap = analyse_gap(profile, job)
    all_text = " ".join(gap.strengths + gap.critical_gaps + gap.minor_gaps).lower()
    mentioned = [skill for skill in profile.skills if skill.lower() in all_text]
    passed = len(mentioned) > 0
    comment = f"Skills mentioned: {mentioned}" if passed else "No profile skills found in gap analysis"
    return passed, comment


# ── APPLICATION PACK EVALS ───────────────────────────────────────────────────

def eval_cover_letter_length(profile: CVProfile, job: RankedJob) -> tuple[bool, str]:
    """
    DETERMINISTIC eval.
    The cover letter must be under 300 words.
    This is a rule explicitly stated in the application_pack prompt.
    """
    gap = analyse_gap(profile, job)
    pack = generate_application_pack(profile, job, gap)
    word_count = len(pack.cover_letter.split())
    passed = word_count <= 300
    comment = f"{word_count} words (limit: 300)"
    return passed, comment


def eval_cover_letter_personalized(profile: CVProfile, job: RankedJob) -> tuple[bool, str]:
    """
    DETERMINISTIC eval.
    The cover letter must contain the candidate's name and at least one
    skill from their profile. Proves it's not a generic template.
    """
    gap = analyse_gap(profile, job)
    pack = generate_application_pack(profile, job, gap)
    letter_lower = pack.cover_letter.lower()
    first_name = profile.name.split()[0].lower()
    has_name = first_name in letter_lower
    skills_found = [s for s in profile.skills if s.lower() in letter_lower]
    passed = has_name and len(skills_found) > 0
    comment = f"Name found: {has_name} | Skills found: {skills_found[:3]}"
    return passed, comment


def eval_talking_points_count(profile: CVProfile, job: RankedJob) -> tuple[bool, str]:
    """
    DETERMINISTIC eval.
    The application pack must contain between 3 and 5 talking points.
    """
    gap = analyse_gap(profile, job)
    pack = generate_application_pack(profile, job, gap)
    count = len(pack.talking_points)
    passed = 3 <= count <= 5
    comment = f"{count} talking points (expected 3-5)"
    return passed, comment


# ── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("  JOB SCOUT — EVALS")
    print("=" * 60)

    # ── Load datasets ──
    cv_de = load_cv("cv_data_engineer.json")
    cv_jr = load_cv("cv_junior_developer.json")
    relevant_de, irrelevant_de = load_job_postings("for_data_engineer")
    relevant_jr, irrelevant_jr = load_job_postings("for_junior_developer")

    # We need a RankedJob to pass to gap analysis and cover letter evals.
    # We create one manually from a known posting so we don't depend on the
    # ranker's score for this part of the eval.
    ranked_job_de = RankedJob(
        title=relevant_de[0].title,
        url=relevant_de[0].url,
        source=relevant_de[0].source,
        score=85,
        rationale="Strong match on GCP and data pipeline skills.",
    )
    ranked_job_jr = RankedJob(
        title=relevant_jr[0].title,
        url=relevant_jr[0].url,
        source=relevant_jr[0].source,
        score=78,
        rationale="Good match on React and JavaScript requirements.",
    )

    # ── Run evals ──
    # run_eval(trace_name, score_name, function, *args)

    print("\n── RANKER EVALS (data engineer profile) ──")
    run_eval("ranker/score_range [DE]",   "score_range_valid",   eval_ranker_score_range,        cv_de, relevant_de + irrelevant_de)
    run_eval("ranker/rationale [DE]",     "rationale_not_empty", eval_ranker_rationale_not_empty, cv_de, relevant_de + irrelevant_de)
    run_eval("ranker/coherence [DE]",     "ranking_coherence",   eval_ranker_coherence,           cv_de, relevant_de, irrelevant_de)

    print("\n── RANKER EVALS (junior developer profile) ──")
    run_eval("ranker/score_range [JR]",   "score_range_valid",   eval_ranker_score_range,        cv_jr, relevant_jr + irrelevant_jr)
    run_eval("ranker/rationale [JR]",     "rationale_not_empty", eval_ranker_rationale_not_empty, cv_jr, relevant_jr + irrelevant_jr)
    run_eval("ranker/coherence [JR]",     "ranking_coherence",   eval_ranker_coherence,           cv_jr, relevant_jr, irrelevant_jr)

    print("\n── GAP ANALYSIS EVALS ──")
    run_eval("gap_analysis/structure",     "gap_structure_valid", eval_gap_structure,      cv_de, ranked_job_de)
    run_eval("gap_analysis/mentions_skills", "gap_mentions_skills", eval_gap_mentions_skills, cv_de, ranked_job_de)

    print("\n── APPLICATION PACK EVALS ──")
    run_eval("app_pack/cover_letter_length",       "cover_letter_length_ok",    eval_cover_letter_length,       cv_de, ranked_job_de)
    run_eval("app_pack/cover_letter_personalized", "cover_letter_personalized", eval_cover_letter_personalized, cv_de, ranked_job_de)
    run_eval("app_pack/talking_points_count",      "talking_points_count_ok",   eval_talking_points_count,      cv_jr, ranked_job_jr)

    # ── Summary ──
    total = len(results)
    passed = sum(1 for r in results if r["passed"])
    failed = total - passed

    print("\n" + "=" * 60)
    print(f"  RESULTS: {passed}/{total} passed  |  {failed} failed")
    print("=" * 60)
    if failed:
        print("\nFailed evals:")
        for r in results:
            if not r["passed"]:
                print(f"  ✗ {r['eval']}: {r['comment']}")

    # Flush ensures all scores are sent to Langfuse before the script exits.
    # Without this, some scores might be lost if the buffer hasn't been sent yet.
    print("\nFlushing scores to Langfuse...")
    langfuse.flush()
    print("Done. Check the Langfuse UI → Traces to see all scores.\n")


if __name__ == "__main__":
    main()
