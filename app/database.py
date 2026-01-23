import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

DB_PATH = Path("conversation_memory.db")

def init_db():
    """Initialize the SQLite database and create the table if it doesn't exist."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create table with the specified schema
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS conversations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        user_query TEXT,
        agent_answer TEXT,
        agent_thinking TEXT,
        query_answer_context TEXT,
        cumilative_context TEXT,
        timestamp TEXT,
        input_audio_path TEXT,
        output_audio_path TEXT
    )
    ''')
    
    conn.commit()
    conn.close()

def get_cumulative_context(session_id: str) -> str:
    """
    Retrieve the cumulative context for the given session_id.
    It fetches the last interaction to construct the context history.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT cumilative_context, user_query, agent_answer 
        FROM conversations 
        WHERE session_id = ? 
        ORDER BY id DESC 
        LIMIT 1
        ''', (session_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return ""
        
        # row = (prev_cumilative_context, prev_user_query, prev_agent_answer)
        prev_context = row[0] or ""
        prev_query = row[1] or ""
        prev_answer = row[2] or ""
        
        # Construct new context
        # format: 
        # Human: ...
        # AI: ...
        
        new_entry = f"Human: {prev_query}\nAI: {prev_answer}\n"
        
        if prev_context:
            return f"{prev_context}\n{new_entry}"
        else:
            return new_entry
    except Exception as e:
        print(f"Error retrieving context: {e}")
        return ""

def save_interaction(
    session_id: str,
    user_query: str,
    agent_answer: str,
    agent_thinking: str,
    query_answer_context: str,
    cumilative_context: str,
    input_audio_path: Optional[str] = None,
    output_audio_path: Optional[str] = None
):
    """Save the interaction to the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    timestamp = datetime.now().isoformat()
    
    cursor.execute('''
    INSERT INTO conversations (
        session_id, user_query, agent_answer, agent_thinking, 
        query_answer_context, cumilative_context, timestamp, 
        input_audio_path, output_audio_path
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        session_id, user_query, agent_answer, agent_thinking,
        query_answer_context, cumilative_context, timestamp,
        input_audio_path, output_audio_path
    ))
    
    conn.commit()
    conn.close()

# Initialize on module load temporarily or call explicit init
init_db()

