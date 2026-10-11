"""
Ranker Agent

Takes a CVProfile and a list of JobPostings,
scores each job 0-100 for fit, and returns them sorted best-first.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langfuse import observe
from pydantic import BaseModel

from agents.cv_parser import CVProfile
from agents.job_search import JobPosting

load_dotenv()

MODEL = os.getenv("MODEL", "openai/gpt-oss-120b")
API_KEY = os.getenv("GROQ_API_KEY")
API_BASE = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")

if not API_KEY:
    raise EnvironmentError("GROQ_API_KEY not found. Check your .env file.")

llm = ChatOpenAI(
    model=MODEL,
    api_key=API_KEY,
    base_url=API_BASE,
    temperature=0,
)


# --- Output structures ---

# JobScore: what the LLM outputs — only score and rationale.
# We don't ask the LLM to repeat title/url/source: we already have them
# from the JobPosting, and asking the LLM to repeat them risks hallucinations.
class JobScore(BaseModel):
    score: int      # 0-100
    rationale: str  # honest explanation of the score


# RankedJob: the final object we return — combines JobPosting data + LLM score.
class RankedJob(BaseModel):
    title: str
    url: str
    source: str
    score: int
    rationale: str


# --- Parser ---
parser = PydanticOutputParser(pydantic_object=JobScore)


# --- Prompt ---
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert recruiter. Score the fit of the candidate for the job from 0 to 100:\n"
        "- 90-100: near-perfect match\n"
        "- 70-89: strong match, minor gaps\n"
        "- 50-69: partial match, some meaningful gaps\n"
        "- 30-49: weak match, significant gaps\n"
        "- 0-29: poor match\n\n"
        "Be honest about gaps. Do not flatter.\n\n"
        "{format_instructions}",
    ),
    (
        "user",
        "CANDIDATE PROFILE:\n{profile_text}\n\n"
        "JOB POSTING:\n"
        "Title: {job_title}\n"
        "Source: {job_source}\n"
        "Description:\n{job_description}",
    ),
])


# --- Chain ---
chain = prompt | llm | parser


# --- Main functions ---

@observe()
def rank_job(profile: CVProfile, job: JobPosting) -> RankedJob:
    """
    Scores a single job posting against the candidate profile.
    """
    profile_text = (
        f"Name: {profile.name}\n"
        f"Skills: {', '.join(profile.skills)}\n"
        f"Experience: {'; '.join(f'{e.role} at {e.company} ({e.duration})' for e in profile.experience)}\n"
        f"Education: {'; '.join(f'{e.degree} from {e.institution}' for e in profile.education)}\n"
        f"Languages: {', '.join(profile.languages)}"
    )

    job_score = chain.invoke({
        "profile_text": profile_text,
        "job_title": job.title,
        "job_source": job.source,
        "job_description": job.description,
        "format_instructions": parser.get_format_instructions(),
    })

    return RankedJob(
        title=job.title,
        url=job.url,
        source=job.source,
        score=job_score.score,
        rationale=job_score.rationale,
    )


@observe()
def rank_jobs(profile: CVProfile, jobs: list[JobPosting]) -> list[RankedJob]:
    """
    Scores all job postings and returns them sorted best-first.
    """
    print(f"[ranker] Scoring {len(jobs)} jobs...")

    ranked = []
    for i, job in enumerate(jobs, 1):
        print(f"[ranker] Scoring job {i}/{len(jobs)}: {job.title}")
        result = rank_job(profile, job)
        ranked.append(result)

    ranked.sort(key=lambda j: j.score, reverse=True)
    return ranked


# --- Entry point ---

if __name__ == "__main__":
    from agents.cv_parser import parse_cv
    from agents.job_search import search_jobs

    if len(sys.argv) < 2:
        print("Usage: python agents/ranker.py <path_cv.pdf>")
        sys.exit(1)

    profile = parse_cv(sys.argv[1])
    jobs = search_jobs(profile)
    ranked = rank_jobs(profile, jobs)

    print("\n--- RANKED JOBS ---")
    for job in ranked:
        print(f"\n[{job.score}/100] {job.title}")
        print(f"  Source: {job.source}")
        print(f"  {job.rationale}")
        print(f"  URL: {job.url}")
