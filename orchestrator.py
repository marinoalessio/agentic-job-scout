"""
Orchestrator

Coordinates the full pipeline and manages the memory lifecycle.

New LangChain concepts:
  - RunnableLambda: wraps any Python function as a chain component
  - RunnableBranch: conditional routing — if condition → chain A, else → chain B

The pipeline state flows as a dict through each step:
  {"profile", "location"}
    → search_step → adds "jobs"
    → rank_step   → adds "ranked"
    → retry_branch → if score low: reformulate → search → rank
                   → else: passthrough
"""

import os
import sys

from dataclasses import dataclass

from langchain_core.runnables import RunnableBranch, RunnableLambda

from agents.cv_parser import CVProfile, parse_cv
from agents.job_search import JobPosting, search_jobs
from agents.query_reformulator import reformulate_query
from agents.ranker import RankedJob, rank_jobs
from memory import session_store, user_prefs
from tracing import langfuse_handler

SCORE_THRESHOLD = 60


# --- Short-term memory ---

@dataclass
class SessionState:
    profile: CVProfile
    location: str
    jobs: list[JobPosting]
    ranked_jobs: list[RankedJob]


# --- Location resolution ---

def resolve_location(profile: CVProfile) -> str:
    saved = user_prefs.get("target_location")
    if saved:
        print(f"[orchestrator] Using saved location preference: {saved}")
        return saved

    if profile.target_location:
        location = profile.target_location
        print(f"[orchestrator] Using target location from CV: {location}")
    elif profile.location:
        location = profile.location
        print(f"[orchestrator] Using current location from CV: {location}")
    else:
        location = input("No location found. Which city/country are you looking for jobs in? ").strip()

    user_prefs.set("target_location", location)
    print(f"[orchestrator] Location '{location}' saved for future runs.")
    return location


# --- Pipeline steps as RunnableLambda ---
# Each step receives the full state dict and returns it enriched with new keys.
# RunnableLambda turns any function into a chain component
# that can be connected with | like any other LangChain runnable.

def _search(state: dict) -> dict:
    jobs = search_jobs(
        state["profile"],
        location=state["location"],
        custom_query=state.get("custom_query", ""),
    )
    # On retry, exclude URLs already seen in the first search
    seen = state.get("seen_urls", set())
    if seen:
        jobs = [j for j in jobs if j.url not in seen]
        print(f"[orchestrator] After deduplication: {len(jobs)} new jobs.")
    state["jobs"] = jobs
    return state


def _rank(state: dict) -> dict:
    state["ranked"] = rank_jobs(state["profile"], state["jobs"])
    return state


def _reformulate(state: dict) -> dict:
    """Generates a new query and stores it in state for the next search step."""
    print(f"\n[orchestrator] Best score {state['ranked'][0].score}/100 is below {SCORE_THRESHOLD}. Reformulating...\n")
    original_query = f"{state['profile'].experience[0].role} job openings in {state['location']}"
    new_query = reformulate_query(original_query, state["profile"], state["ranked"])
    print(f"[orchestrator] New query: '{new_query}'")
    state["custom_query"] = new_query
    # Save seen URLs so _search can exclude duplicates on retry
    state["seen_urls"] = {job.url for job in state["ranked"]}
    # Save previous results to compare later
    state["original_ranked"] = state["ranked"]
    return state


def _keep_best(state: dict) -> dict:
    """After retry, keeps whichever attempt had the better top score."""
    original = state.get("original_ranked", [])
    if original and original[0].score >= state["ranked"][0].score:
        print("[orchestrator] Reformulation did not improve results. Keeping original.")
        state["ranked"] = original
    else:
        print(f"[orchestrator] Improved results (best score: {state['ranked'][0].score}/100). Using new results.")
    return state


# --- Chain assembly ---

search_and_rank = RunnableLambda(_search) | RunnableLambda(_rank)

retry_chain = (
    RunnableLambda(_reformulate)
    | RunnableLambda(_search)
    | RunnableLambda(_rank)
    | RunnableLambda(_keep_best)
)

# RunnableBranch takes a list of (condition, chain) tuples.
# It evaluates conditions in order and runs the first matching chain.
# The last argument is the default (passthrough when no condition matches).
def _should_retry(state: dict) -> bool:
    result = bool(state["ranked"]) and state["ranked"][0].score < SCORE_THRESHOLD
    print(f"[orchestrator] Best score: {state['ranked'][0].score if state['ranked'] else 'N/A'} — retry: {result}")
    return result

retry_branch = RunnableBranch(
    (_should_retry, retry_chain),
    RunnableLambda(lambda state: state),
)

pipeline = search_and_rank | retry_branch


# --- Main function ---

def run(pdf_path: str) -> SessionState:
    print("\n=== JOB SCOUT — Starting pipeline ===\n")

    profile = parse_cv(pdf_path)
    location = resolve_location(profile)

    # Invoke the full LCEL pipeline with the initial state
    final_state = pipeline.invoke(
        {"profile": profile, "location": location},
        config={"callbacks": [langfuse_handler]},
    )

    session_store.save_session(
        candidate_name=profile.name,
        location=location,
        ranked_jobs=final_state["ranked"],
    )

    return SessionState(
        profile=profile,
        location=location,
        jobs=final_state["jobs"],
        ranked_jobs=final_state["ranked"],
    )


# --- Entry point ---

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python orchestrator.py <path_cv.pdf>")
        sys.exit(1)

    state = run(sys.argv[1])

    print(f"\n=== RESULTS for {state.profile.name} — {state.location} ===\n")
    for job in state.ranked_jobs:
        print(f"[{job.score}/100] {job.title}")
        print(f"  {job.rationale}")
        print(f"  {job.url}\n")
