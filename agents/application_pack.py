"""
Application Pack Agent

Takes a CVProfile, a RankedJob, and a GapAnalysis and generates
a tailored cover letter and interview talking points.

It uses the gap analysis as input so the cover letter can emphasise
real strengths and prepare the candidate for likely questions about gaps.

Input:  CVProfile + RankedJob + GapAnalysis
Output: ApplicationPack (cover_letter, talking_points)
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
from agents.gap_analysis import GapAnalysis
from agents.ranker import RankedJob
from tracing import langfuse_handler

load_dotenv()

MODEL = os.getenv("MODEL", "openai/gpt-oss-120b")
API_KEY = os.getenv("GROQ_API_KEY")
API_BASE = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")

llm = ChatOpenAI(model=MODEL, api_key=API_KEY, base_url=API_BASE, temperature=0.3)

# temperature=0.3: we want some natural variation in writing style,
# but not so much creativity that the letter becomes inaccurate.


# --- Output structure ---

class ApplicationPack(BaseModel):
    job_title: str
    cover_letter: str       # full cover letter, ready to send
    talking_points: list[str]   # 3-5 key points to use during an interview


# --- Chain ---

parser = PydanticOutputParser(pydantic_object=ApplicationPack)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert career coach who writes tailored, specific cover letters.\n"
        "Rules:\n"
        "- Reference real skills and experiences from the candidate's profile\n"
        "- Address the job requirements directly\n"
        "- Do NOT mention the gaps — the cover letter should focus on strengths\n"
        "- Talking points should help the candidate address both strengths AND gaps in an interview\n"
        "- Keep the cover letter under 300 words\n"
        "- Write in a professional but human tone — not robotic\n\n"
        "{format_instructions}",
    ),
    (
        "user",
        "CANDIDATE PROFILE:\n{profile_text}\n\n"
        "JOB:\n"
        "Title: {job_title}\n"
        "URL: {job_url}\n\n"
        "GAP ANALYSIS (use this to inform the letter, but don't mention gaps directly):\n"
        "Strengths: {strengths}\n"
        "Critical gaps: {critical_gaps}\n"
        "Minor gaps: {minor_gaps}",
    ),
])

chain = prompt | llm | parser


# --- Main function ---

@observe()
def generate_application_pack(
    profile: CVProfile,
    job: RankedJob,
    gap: GapAnalysis,
) -> ApplicationPack:
    """
    Generates a cover letter and talking points for a specific job.
    """
    print(f"[application_pack] Generating pack for: {job.title}")

    profile_text = (
        f"Name: {profile.name}\n"
        f"Skills: {', '.join(profile.skills)}\n"
        f"Experience: {'; '.join(f'{e.role} at {e.company} ({e.duration}): {e.description}' for e in profile.experience)}\n"
        f"Education: {'; '.join(f'{e.degree} from {e.institution}' for e in profile.education)}\n"
        f"Languages: {', '.join(profile.languages)}"
    )

    pack = chain.invoke(
        {
            "profile_text": profile_text,
            "job_title": job.title,
            "job_url": job.url,
            "strengths": "\n".join(f"- {s}" for s in gap.strengths),
            "critical_gaps": "\n".join(f"- {g}" for g in gap.critical_gaps),
            "minor_gaps": "\n".join(f"- {g}" for g in gap.minor_gaps),
            "format_instructions": parser.get_format_instructions(),
        },
        config={"callbacks": [langfuse_handler]},
    )

    return pack
