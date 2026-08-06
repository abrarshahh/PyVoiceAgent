"""Response generation for the single executor pipeline.

This module replaced `app/workflows/graph.py` + `app/agents/response_agent.py`.
It's a plain function, not a LangGraph node.
"""
import re
import emoji
from edgevoice.core.llm import get_llm
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from edgevoice.core.logging import get_logger
from edgevoice.core.db import get_cumulative_context

logger = get_logger(__name__)

# Lazy-loaded LLM
_llm = None

def get_assistant_llm():
    global _llm
    if _llm is None:
        try:
            _llm = get_llm(json_mode=False)
        except NotImplementedError:
            _llm = None
    return _llm


def generate_response(
    text: str,
    session_id: str | None = None,
    execution_log: list | None = None,
    execution_mode: str = "chat",
) -> dict:
    """Generate a textual response for a user turn.

    Returns a dict with keys: response_text, agent_thinking, past_context.
    """
    execution_log = execution_log or []

    # Retrieve persistent context from SQLite
    past_context = ""
    if session_id:
        past_context = get_cumulative_context(session_id)

    # Construct system prompt with history
    system_content = (
        "You are a helpful voice assistant. Keep your responses concise and conversational. "
        "IMPORTANT: You must format your final response entirely in UPPERCASE letters. "
        "Use clear sentence boundaries."
    )
    if past_context:
        system_content += f"\n\nPrevious conversation history:\n{past_context}"

    system_message = SystemMessage(content=system_content)
    human_msg = HumanMessage(content=text)

    prompt_messages = [system_message, human_msg]

    # Invoke the local LLM
    try:
        assistant_llm = get_assistant_llm()
        if assistant_llm is None:
            raise NotImplementedError("Assistant LLM is not configured/implemented.")
        response = assistant_llm.invoke(prompt_messages)
    except Exception as e:
        logger.error(f"LLM invocation failed: {e}")
        error_msg = "I APOLOGIZE, BUT I AM HAVING TROUBLE THINKING RIGHT NOW."
        return {
            "response_text": error_msg,
            "agent_thinking": f"Error: {str(e)}",
            "past_context": past_context,
        }

    # Filter out <think>...</think> tags from DeepSeek R1
    raw_content = response.content
    think_match = re.search(r"<think>(.*?)</think>", raw_content, flags=re.DOTALL)
    agent_thinking = think_match.group(1).strip() if think_match else ""

    content = re.sub(r"<think>.*?</think>", "", raw_content, flags=re.DOTALL).strip()
    content = emoji.replace_emoji(content, replace="")

    logger.agent_output(f"Agent Response: {content}")

    return {
        "response_text": content,
        "agent_thinking": agent_thinking,
        "past_context": past_context,
    }
