import re
import emoji
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from app.workflows.state import AgentState
from app.core.logging import get_logger
from app.db.storage import get_cumulative_context

logger = get_logger(__name__)

# Initialize Local LLM (DeepSeek R1 via Ollama)
llm = ChatOllama(model="deepseek-r1:8b")

def process_input(state: AgentState) -> AgentState:
    """Node to process text input and generate a textual response."""
    text = state.get("input_text", "")
    session_id = state.get("session_id")
    
    logger.info("Processing input text.")
    logger.agent_output(f"User Input: {text}")
    
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
        response = llm.invoke(prompt_messages)
    except Exception as e:
        logger.error(f"LLM invocation failed: {e}")
        error_msg = "I APOLOGIZE, BUT I AM HAVING TROUBLE THINKING RIGHT NOW."
        return {
            "response_text": error_msg,
            "messages": [human_msg, AIMessage(content=error_msg)],
            "agent_thinking": f"Error: {str(e)}",
            "cumilative_context": past_context
        }
    
    # Filter out <think>...</think> tags from DeepSeek R1
    raw_content = response.content
    agent_thinking = ""
    
    # Extract thinking
    think_match = re.search(r'<think>(.*?)</think>', raw_content, flags=re.DOTALL)
    if think_match:
        agent_thinking = think_match.group(1).strip()
    
    content = re.sub(r'<think>.*?</think>', '', raw_content, flags=re.DOTALL).strip()
    
    # Remove emojis
    content = emoji.replace_emoji(content, replace='')
    
    # Update the response content with cleaned text
    response.content = content
    
    logger.agent_output(f"Agent Response: {content}")
    
    # Return updates
    return {
        "response_text": content, 
        "messages": [human_msg, response],
        "agent_thinking": agent_thinking,
        "cumilative_context": past_context
    }

