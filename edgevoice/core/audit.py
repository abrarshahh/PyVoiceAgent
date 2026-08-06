"""Conversation persistence.

Refactored from a LangGraph node into a plain function.
"""
import re
from edgevoice.core.llm import get_llm
from langchain_core.messages import HumanMessage
from edgevoice.core.logging import get_logger
from edgevoice.core.db import save_interaction

logger = get_logger(__name__)

# Lazy-loaded LLM for summarization
_llm = None

def get_summarizer():
    global _llm
    if _llm is None:
        try:
            _llm = get_llm(json_mode=False)
        except NotImplementedError:
            _llm = None
    return _llm


def save_conversation(
    session_id: str,
    user_query: str,
    agent_answer: str,
    agent_thinking: str = "",
    cumilative_context: str = "",
    input_audio_path: str | None = None,
    output_audio_path: str | None = None,
) -> str:
    """Summarize the turn and write it to the DB. Returns the summary text."""
    if not (session_id and user_query and agent_answer):
        return ""

    summary_prompt = f"""Summarize the following interaction concisely in one sentence.

User: {user_query}
Agent: {agent_answer}

Summary:"""

    query_answer_context = ""
    try:
        summarizer = get_summarizer()
        if summarizer is None:
            raise NotImplementedError("Summarizer LLM is not configured/implemented.")
        summary_response = summarizer.invoke([HumanMessage(content=summary_prompt)])
        raw_summary = summary_response.content
        query_answer_context = re.sub(
            r"<think>.*?</think>", "", raw_summary, flags=re.DOTALL
        ).strip()
        logger.info("Generated interaction summary.")
    except Exception as e:
        logger.error(f"Failed to generate summary: {e}")
        query_answer_context = f"User asked about {user_query[:20]}..."

    try:
        save_interaction(
            session_id=session_id,
            user_query=user_query,
            agent_answer=agent_answer,
            agent_thinking=agent_thinking,
            query_answer_context=query_answer_context,
            cumilative_context=cumilative_context,
            input_audio_path=input_audio_path,
            output_audio_path=output_audio_path,
        )
        logger.info(f"Interaction saved for session {session_id}")
    except Exception as e:
        logger.error(f"Failed to save interaction: {e}")

    return query_answer_context
