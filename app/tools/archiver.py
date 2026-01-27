import re
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from app.workflows.state import AgentState
from app.core.logging import get_logger
from app.db.storage import save_interaction

logger = get_logger(__name__)

# Initialize separate LLM instance for summarization
llm = ChatOllama(model="deepseek-r1:8b")

def save_conversation(state: AgentState) -> AgentState:
    """Node to save the interaction to the database."""
    session_id = state.get("session_id")
    user_query = state.get("input_text")
    agent_answer = state.get("response_text")
    agent_thinking = state.get("agent_thinking", "")
    cumilative_context = state.get("cumilative_context", "")
    input_audio = state.get("input_audio_path")
    output_audio = state.get("response_audio_path")
    
    if session_id and user_query and agent_answer:
        # Generate summary of this interaction
        summary_prompt = f"""Summarize the following interaction concisely in one sentence.
        
        User: {user_query}
        Agent: {agent_answer}
        
        Summary:"""
        
        query_answer_context = ""
        try:
            summary_response = llm.invoke([HumanMessage(content=summary_prompt)])
            # Clean up thinking tags if any (DeepSeek R1)
            raw_summary = summary_response.content
            query_answer_context = re.sub(r'<think>.*?</think>', '', raw_summary, flags=re.DOTALL).strip()
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
                input_audio_path=input_audio,
                output_audio_path=output_audio
            )
            logger.info(f"Interaction saved for session {session_id}")
        except Exception as e:
            logger.error(f"Failed to save interaction: {e}")
            
    return {"query_answer_context": query_answer_context}

