"""
Query Reformulator Agent

When ranked results have low scores, this agent uses an LLM to analyse
what went wrong and generate a better search query.

New LangChain concept: StrOutputParser
  - Like PydanticOutputParser but returns a plain string instead of a Pydantic object.
  - Use it when you just need text from the LLM, not structured data.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from agents.cv_parser import CVProfile
from agents.ranker import RankedJob
from tracing import langfuse_handler

load_dotenv()

MODEL = os.getenv("MODEL", "openai/gpt-oss-120b")
API_KEY = os.getenv("GROQ_API_KEY")
API_BASE = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")

llm = ChatOpenAI(model=MODEL, api_key=API_KEY, base_url=API_BASE, temperature=0.3)

# temperature=0.3 here (not 0) because we want some creativity
# in reformulating — a deterministic reformulation of a bad query
# would likely produce another bad query.

# StrOutputParser: just returns the LLM response as a plain string.
# No JSON, no schema — the LLM output IS the new query.
parser = StrOutputParser()

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a job search expert. A search query returned poor results — "
        "the job postings found were not relevant to the candidate.\n\n"
        "Your task: generate ONE improved search query that will find better results.\n"
        "Rules:\n"
        "- Return ONLY the search query string, nothing else\n"
        "- Make it more specific or try different keywords\n"
        "- Keep the location if present\n"
        "- Do not add quotes or special characters",
    ),
    (
        "user",
        "Original query: {original_query}\n\n"
        "Candidate skills: {skills}\n\n"
        "Why results were poor (top 3 jobs and their scores):\n{poor_results_summary}",
    ),
])

chain = prompt | llm | parser


def reformulate_query(
    original_query: str,
    profile: CVProfile,
    ranked_jobs: list[RankedJob],
) -> str:
    """
    Generates a better search query when results are unsatisfactory.

    Takes the original query, the candidate profile, and the poor results
    so the LLM understands what went wrong.
    """
    # Summarise the top 3 results so the LLM sees what we got
    top_3 = ranked_jobs[:3]
    poor_results_summary = "\n".join(
        f"- [{job.score}/100] {job.title} ({job.source}): {job.rationale}"
        for job in top_3
    )

    new_query = chain.invoke(
        {
            "original_query": original_query,
            "skills": ", ".join(profile.skills[:6]),
            "poor_results_summary": poor_results_summary,
        },
        config={"callbacks": [langfuse_handler]},
    )

    return new_query.strip()
