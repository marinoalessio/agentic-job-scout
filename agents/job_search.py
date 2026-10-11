"""
Job Search Agent

Takes a CVProfile, builds a search query from it,
and uses Tavily to find real job postings on the web.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from langfuse import observe
from pydantic import BaseModel

from agents.cv_parser import CVProfile

load_dotenv()

# --- Tool ---
# TavilySearchResults is a pre-built LangChain tool.
# A tool has: a name, a description (used by agents to decide when to call it),
# and a defined input schema.
# Here we call it manually with .invoke() — in future phases
# an agent will call it autonomously.
search_tool = TavilySearch(
    max_results=10,   # fetch up to 10; callers can slice to fewer
    api_key=os.getenv("TAVILY_API_KEY"),
)


# --- Output structure ---

class JobPosting(BaseModel):
    title: str
    url: str
    description: str    # raw text from Tavily — will be used by the ranker
    source: str         # domain name, e.g. "linkedin.com"


# --- Query builder ---

def build_search_query(profile: CVProfile, location: str) -> str:
    """
    Builds a plain-english search query from the CV profile and location.
    Location is resolved upstream by the orchestrator.
    """
    if profile.experience:
        role = profile.experience[0].role
    else:
        role = "professional"

    skills_snippet = ", ".join(profile.skills[:4]) if profile.skills else ""

    query = f"{role} job openings"
    if skills_snippet:
        query += f" {skills_snippet}"
    if location:
        query += f" in {location}"

    return query


# --- Main function ---

@observe()
def search_jobs(profile: CVProfile, location: str = "", custom_query: str = "", max_results: int = 5) -> list[JobPosting]:
    """
    Searches for job postings matching the profile using the Tavily tool.

    If custom_query is provided (e.g. from the reformulator), it overrides
    the auto-generated query. This allows the orchestrator to retry with
    a better query without changing this agent's logic.

    Returns a list of JobPosting objects ready to be passed to the ranker.
    """
    query = custom_query if custom_query else build_search_query(profile, location)
    print(f"[job_search] Query: '{query}'")

    # .invoke() is the standard way to call any LangChain tool.
    # It takes a dict with the input — here {"query": "..."}
    # and returns a list of dicts: [{"url": "...", "content": "...", "title": "..."}, ...]
    response = search_tool.invoke({"query": query})
    raw_results = response["results"]
    print(f"[job_search] Found {len(raw_results)} results.")

    postings = []
    for r in raw_results:
        url = r.get("url", "")
        posting = JobPosting(
            title=r.get("title", ""),
            url=url,
            description=r.get("content", ""),
            source=url.split("/")[2] if url else "",
        )
        postings.append(posting)

    return postings[:max_results]


# --- Entry point ---

if __name__ == "__main__":
    from agents.cv_parser import parse_cv

    if len(sys.argv) < 2:
        print("Usage: python agents/job_search.py <path_cv.pdf>")
        sys.exit(1)

    profile = parse_cv(sys.argv[1])
    postings = search_jobs(profile)

    print("\n--- JOB POSTINGS FOUND ---")
    for i, job in enumerate(postings, 1):
        print(f"\n[{i}] {job.title}")
        print(f"    Source: {job.source}")
        print(f"    URL: {job.url}")
        print(f"    Preview: {job.description[:150]}...")
