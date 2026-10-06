"""
CV Parser Agent

It takes a PDF file, extract the text, and then asks an LLM to structure it
in a JSON profile: name, skills, work experience and education.
"""

import json
import os
import sys

# Add project root to path so that "tools" module can be found
# regardless of where the script is launched from
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

from tools.pdf_extractor import extract_text_from_pdf

# Load variables from .env
load_dotenv()

MODEL = os.getenv("MODEL", "openai/gpt-oss-120b")
API_KEY = os.getenv("GROQ_API_KEY")
API_BASE = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")

if not API_KEY:
    raise EnvironmentError("GROQ_API_KEY not found. Check your .env file.")

# OpenAI client pointed at Groq
client = OpenAI(api_key=API_KEY, base_url=API_BASE)


# --- Structure of the output ---
# Pydantic defines the "schema" of the profile you want to obtain.
# It's like a dataclass with an automatic validation of the types
class Experience(BaseModel):
    company: str
    role: str
    duration: str       # ex. "2021–2023"
    description: str    # brief summary of the experience


class Education(BaseModel):
    institution: str
    degree: str
    year: str


class CVProfile(BaseModel):
    name: str
    email: str
    skills: list[str]           # list of technical skills
    experience: list[Experience]
    education: list[Education]
    languages: list[str]        # spoken languages


# --- The prompt to send to the LLM ---
SYSTEM_PROMPT = """
You are a specialized assistant in resume analysis.
You will be given a text from a resume.
Your job is to extract key information and return them ONLY as a valid JSON,
with no extra text, no markdown, no explanations.
The JSON must respect this schema:

{
  "name": "Name Surname",
  "email": "email@example.com",
  "skills": ["Python", "SQL", "..."],
  "experience": [
    {
      "company": "Company Name",
      "role": "Role Title",
      "duration": "oct 2021 - nov 2023",
      "description": "Summary of the experience"
    }
  ],
  "education": [
    {
      "institution": "University Name",
      "degree": "Master's degree in Computer Science",
      "year": "2020"
    }
  ],
  "languages": ["Italian", "English"]
}

If a field is not present in the resume, use an empty string "" or an empty list []
"""


# --- Main function of the agent ---

def parse_cv(pdf_path: str) -> CVProfile:
    """
    It takes a path of a PDF file and returns a structured CVProfile object

    Flow:
    1. Extract text from PDF
    2. Send the text to the LLM with instructions
    3. Parse the JSON response to a CVProfile
    """
    print(f"[cv_parser] Extracting text from: {pdf_path}")
    raw_text = extract_text_from_pdf(pdf_path)

    if not raw_text.strip():
        raise ValueError("The PDF does not contain a valid text.")

    print(f"[cv_parser] Text extracted ({len(raw_text)} characters). Sending to LLM...")

    # Call to the LLM — identical to what n8n does via HTTP
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Here the text of the resume:\n\n{raw_text}"},
        ],
        temperature=0,      # 0 = deterministic, no creativity.
    )

    # Extract response
    raw_json = response.choices[0].message.content
    print("[cv_parser] Response received. Parsing JSON...")

    # Converts the JSON string to a Python dict, then validated through Pydantic
    data = json.loads(raw_json)
    profile = CVProfile(**data)

    print(f"[cv_parser] Profile extracted for: {profile.name}")
    return profile


# --- Entry point to test it in the terminal ---
# This block only runs if you launch it manually from the terminal:
#   python agents/cv_parser.py my_cv.pdf

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python agents/cv_parser.py <path_cv.pdf>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    profile = parse_cv(pdf_path)

    # Print in a readable text
    print("\n--- EXTRACTED PROFILE ---")
    print(profile.model_dump_json(indent=2))
