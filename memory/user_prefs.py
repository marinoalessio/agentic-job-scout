"""
User Preferences — Long-term memory (Level 3)

Stores persistent user preferences in a SQLite database.
Each preference is a key/value pair that survives across sessions.

"""

import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "job_scout.db")


def _get_connection() -> sqlite3.Connection:
    """Opens a connection to the SQLite database and creates the table if needed."""
    # exist_ok=True avoids errors if the directory already exists
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

    conn = sqlite3.connect(DB_PATH)

    # CREATE TABLE IF NOT EXISTS → safe to call every time, does nothing if table exists
    conn.execute("""
        CREATE TABLE IF NOT EXISTS preferences (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
    """)
    conn.commit()
    return conn


def get(key: str) -> str | None:
    """
    Returns the value for a preference key, or None if not set.

    Example:
        location = get("target_location")  # → "Sydney" or None
    """
    conn = _get_connection()
    row = conn.execute("SELECT value FROM preferences WHERE key = ?", (key,)).fetchone()
    conn.close()
    return row[0] if row else None


def set(key: str, value: str) -> None:
    """
    Saves a preference. Overwrites if it already exists.

    SQLite's INSERT OR REPLACE handles both insert and update in one query.
    """
    conn = _get_connection()
    conn.execute("INSERT OR REPLACE INTO preferences (key, value) VALUES (?, ?)", (key, value))
    conn.commit()
    conn.close()


def delete(key: str) -> None:
    """Removes a preference."""
    conn = _get_connection()
    conn.execute("DELETE FROM preferences WHERE key = ?", (key,))
    conn.commit()
    conn.close()


def all_prefs() -> dict[str, str]:
    """Returns all saved preferences as a dictionary."""
    conn = _get_connection()
    rows = conn.execute("SELECT key, value FROM preferences").fetchall()
    conn.close()
    return {row[0]: row[1] for row in rows}
