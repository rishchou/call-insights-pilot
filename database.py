import sqlite3
import json
import streamlit as st

DB_NAME = "transcripts.db"

def init_db():
    """Initializes the database and creates the transcripts table if it doesn't exist."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS transcripts (
            file_hash TEXT PRIMARY KEY,
            file_name TEXT NOT NULL,
            transcript_data TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

@st.cache_resource
def get_db_connection():
    """Returns a database connection."""
    return sqlite3.connect(DB_NAME, check_same_thread=False)

def save_transcript(file_hash: str, file_name: str, transcript_data: dict):
    """Saves a new transcript record to the database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    # Convert the transcript dictionary to a JSON string for storage
    transcript_json = json.dumps(transcript_data)
    cursor.execute(
        "INSERT INTO transcripts (file_hash, file_name, transcript_data) VALUES (?, ?, ?)",
        (file_hash, file_name, transcript_json)
    )
    conn.commit()

def fetch_transcript(file_hash: str):
    """Fetches an existing transcript from the database by its hash."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT transcript_data FROM transcripts WHERE file_hash = ?", (file_hash,))
    result = cursor.fetchone()
    if result:
        # Convert the JSON string back to a dictionary
        return json.loads(result[0])
    return None
