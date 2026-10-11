"""
Orchestrator

Coordinates the full pipeline and manages the memory lifecycle:
  1. Load user preferences (long-term memory)
  2. Parse CV → CVProfile
  3. Resolve location (from prefs > CV target > CV current > ask user)
  4. Search jobs
  5. Rank jobs
  6. Save session (long-term memory)
  7. Return SessionState (short-term memory)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataclasses import dataclass

from agents.cv_parser import CVProfile, parse_cv
from agents.job_search import JobPosting, search_jobs
from agents.ranker import RankedJob, rank_jobs
from memory import session_store, user_prefs


# --- Short-term memory ---
# SessionState holds everything produced in one pipeline run.
# It lives only in memory for the duration of the run,
# then gets persisted to SQLite by save_session().

@dataclass
class SessionState:
    profile: CVProfile
    location: str
    jobs: list[JobPosting]
    ranked_jobs: list[RankedJob]


# --- Location resolution ---

def resolve_location(profile: CVProfile) -> str:
    """
    Determines where to search for jobs, in order of priority:
    1. Saved user preference (SQLite) — persists across runs
    2. target_location from CV (e.g. "open to relocation to Australia")
    3. current location from CV
    4. Ask the user → save to SQLite for next time
    """
    # 1. Saved preference
    saved = user_prefs.get("target_location")
    if saved:
        print(f"[orchestrator] Using saved location preference: {saved}")
        return saved

    # 2. Relocation preference in the CV
    if profile.target_location:
        location = profile.target_location
        print(f"[orchestrator] Using target location from CV: {location}")

    # 3. Current location in the CV
    elif profile.location:
        location = profile.location
        print(f"[orchestrator] Using current location from CV: {location}")

    # 4. Ask the user
    else:
        location = input("No location found. Which city/country are you looking for jobs in? ").strip()

    # Save in all cases so future runs skip this logic entirely
    user_prefs.set("target_location", location)
    print(f"[orchestrator] Location '{location}' saved for future runs.")
    return location


# --- Main pipeline ---

def run(pdf_path: str) -> SessionState:
    """
    Runs the full pipeline for a given CV.
    Returns a SessionState with all the results.
    """
    print("\n=== JOB SCOUT — Starting pipeline ===\n")

    # Step 1: Parse CV
    profile = parse_cv(pdf_path)

    # Step 2: Resolve location
    location = resolve_location(profile)

    # Step 3: Search jobs
    jobs = search_jobs(profile, location=location)

    # Step 4: Rank jobs
    ranked = rank_jobs(profile, jobs)

    # Step 5: Save session to SQLite
    session_store.save_session(
        candidate_name=profile.name,
        location=location,
        ranked_jobs=ranked,
    )

    # Step 6: Return short-term memory object
    state = SessionState(
        profile=profile,
        location=location,
        jobs=jobs,
        ranked_jobs=ranked,
    )

    return state


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
