"""
Session Store — Long-term memory (Level 2)

Saves the full output of each pipeline run to SQLite.
Each run = one session, with its ranked jobs.

This lets you review past searches and results.
"""

import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "job_scout.db")


def _get_connection() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at     TEXT NOT NULL,
            candidate_name TEXT NOT NULL,
            location       TEXT NOT NULL
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS ranked_jobs (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            title      TEXT NOT NULL,
            url        TEXT NOT NULL,
            source     TEXT NOT NULL,
            score      INTEGER NOT NULL,
            rationale  TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        )
    """)

    conn.commit()
    return conn


def save_session(candidate_name: str, location: str, ranked_jobs: list) -> int:
    """
    Saves a pipeline run to the database.

    Returns the session ID (useful for future reference or linking).
    """
    conn = _get_connection()

    cursor = conn.execute(
        "INSERT INTO sessions (created_at, candidate_name, location) VALUES (?, ?, ?)",
        (datetime.now().isoformat(), candidate_name, location),
    )
    session_id = cursor.lastrowid

    for job in ranked_jobs:
        conn.execute(
            """INSERT INTO ranked_jobs (session_id, title, url, source, score, rationale)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (session_id, job.title, job.url, job.source, job.score, job.rationale),
        )

    conn.commit()
    conn.close()
    print(f"[session_store] Session {session_id} saved ({len(ranked_jobs)} jobs).")
    return session_id


def get_past_sessions() -> list[dict]:
    """Returns a summary of all past sessions."""
    conn = _get_connection()
    rows = conn.execute(
        "SELECT id, created_at, candidate_name, location FROM sessions ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    return [
        {"id": r[0], "created_at": r[1], "candidate_name": r[2], "location": r[3]}
        for r in rows
    ]
