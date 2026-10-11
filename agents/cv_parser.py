"""
CV Parser Agent

It takes a PDF file, extract the text, and then asks an LLM to structure it
in a JSON profile: name, skills, work experience and education.
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

from tools.pdf_extractor import extract_text_from_pdf

load_dotenv()

MODEL = os.getenv("MODEL", "openai/gpt-oss-120b")
API_KEY = os.getenv("GROQ_API_KEY")
API_BASE = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")

if not API_KEY:
    raise EnvironmentError("GROQ_API_KEY not found. Check your .env file.")

# --- LLM ---
# ChatOpenAI is LangChain's wrapper around the OpenAI client.
# We point it to Groq by setting base_url and api_key.
# temperature=0 for deterministic, faithful extraction.
llm = ChatOpenAI(
    model=MODEL,
    api_key=API_KEY,
    base_url=API_BASE,
    temperature=0,
)


# --- Output structure ---

class Experience(BaseModel):
    company: str
    role: str
    duration: str
    description: str


class Education(BaseModel):
    institution: str
    degree: str
    year: str


class CVProfile(BaseModel):
    name: str
    email: str
    location: str           # current city/country of the candidate
    target_location: str    # preferred location for job search (e.g. from "open to relocation to X"), empty string if not specified
    skills: list[str]
    experience: list[Experience]
    education: list[Education]
    languages: list[str]


# --- Parser ---
# PydanticOutputParser reads the CVProfile schema and does two things:
# 1. Generates format instructions to include in the prompt
#    so the LLM knows exactly what JSON to produce
# 2. Parses the LLM response string into a CVProfile object automatically
#    (no more json.loads + CVProfile(**data))
parser = PydanticOutputParser(pydantic_object=CVProfile)


# --- Prompt ---
# ChatPromptTemplate is a reusable prompt with named variables ({}).
# Variables are filled in at call time via chain.invoke({...}).
# {format_instructions} is injected automatically by the parser.
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a specialized assistant in resume analysis.\n"
        "Extract key information from the resume text and return it as structured data.\n"
        "For the location field, extract the city or country where the candidate currently lives.\n"
        "For the target_location field, extract where the candidate wants to work — look for phrases like "
        "'open to relocation to X', 'looking for opportunities in X', 'willing to move to X'. "
        "If not mentioned, use an empty string.\n"
        "If a field is not present in the resume, use an empty string or empty list.\n\n"
        "{format_instructions}",
    ),
    (
        "user",
        "Here is the resume text:\n\n{cv_text}",
    ),
])


# --- Chain ---
# The | operator connects the three pieces into a pipeline:
#   prompt receives variables → formats the messages
#   llm receives the messages → returns a string response
#   parser receives the string → returns a CVProfile object
chain = prompt | llm | parser


# --- Main function ---

@observe()
def parse_cv(pdf_path: str) -> CVProfile:
    """
    Takes a PDF path and returns a structured CVProfile object.

    Flow:
    1. Extract text from PDF
    2. Run the chain: prompt | llm | parser
    3. Return the CVProfile object
    """
    print(f"[cv_parser] Extracting text from: {pdf_path}")
    raw_text = extract_text_from_pdf(pdf_path)

    if not raw_text.strip():
        raise ValueError("The PDF does not contain valid text.")

    print(f"[cv_parser] Text extracted ({len(raw_text)} characters). Running chain...")

    profile = chain.invoke({
        "cv_text": raw_text,
        "format_instructions": parser.get_format_instructions(),
    })

    print(f"[cv_parser] Profile extracted for: {profile.name}")
    return profile


# --- Entry point ---

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python agents/cv_parser.py <path_cv.pdf>")
        sys.exit(1)

    profile = parse_cv(sys.argv[1])

    print("\n--- EXTRACTED PROFILE ---")
    print(profile.model_dump_json(indent=2))
