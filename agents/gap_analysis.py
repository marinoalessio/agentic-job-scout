"""
Gap Analysis Agent

Takes a CVProfile and a RankedJob and produces a detailed, honest analysis
of where the candidate is strong, where they fall short, and whether
it's worth applying.

Input:  CVProfile + RankedJob (already scored by the ranker)
Output: GapAnalysis (strengths, critical gaps, minor gaps, recommendation)
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
from agents.ranker import RankedJob
from tracing import langfuse_handler

load_dotenv()

MODEL = os.getenv("MODEL", "openai/gpt-oss-120b")
API_KEY = os.getenv("GROQ_API_KEY")
API_BASE = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")

llm = ChatOpenAI(model=MODEL, api_key=API_KEY, base_url=API_BASE, temperature=0)


# --- Output structure ---

class GapAnalysis(BaseModel):
    job_title: str
    strengths: list[str]        # what the candidate does well for this role
    critical_gaps: list[str]    # missing skills/experience that would likely block the candidacy
    minor_gaps: list[str]       # gaps that can be compensated or learned quickly
    recommendation: str         # honest 2-3 sentence verdict: apply or not, and why


# --- Chain ---

parser = PydanticOutputParser(pydantic_object=GapAnalysis)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a brutally honest career advisor. "
        "Analyse the fit between a candidate and a specific job posting.\n"
        "Be specific — reference actual skills, tools, and experiences from the CV.\n"
        "Do not be encouraging for its own sake. If the candidate is not ready, say so clearly.\n\n"
        "{format_instructions}",
    ),
    (
        "user",
        "CANDIDATE PROFILE:\n{profile_text}\n\n"
        "JOB:\n"
        "Title: {job_title}\n"
        "Score given by ranker: {score}/100\n"
        "Ranker rationale: {rationale}\n"
        "Job description:\n{job_description}",
    ),
])

chain = prompt | llm | parser


# --- Main function ---

@observe()
def analyse_gap(profile: CVProfile, job: RankedJob) -> GapAnalysis:
    """
    Produces a detailed gap analysis for a single job.
    """
    print(f"[gap_analysis] Analysing gap for: {job.title}")

    profile_text = (
        f"Name: {profile.name}\n"
        f"Skills: {', '.join(profile.skills)}\n"
        f"Experience: {'; '.join(f'{e.role} at {e.company} ({e.duration}): {e.description}' for e in profile.experience)}\n"
        f"Education: {'; '.join(f'{e.degree} from {e.institution}' for e in profile.education)}\n"
        f"Languages: {', '.join(profile.languages)}"
    )

    analysis = chain.invoke(
        {
            "profile_text": profile_text,
            "job_title": job.title,
            "score": job.score,
            "rationale": job.rationale,
            "job_description": f"Source: {job.source}\nURL: {job.url}",
            "format_instructions": parser.get_format_instructions(),
        },
        config={"callbacks": [langfuse_handler]},
    )

    return analysis
