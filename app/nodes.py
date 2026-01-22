import os
import uuid
import re
import emoji
import numpy as np
from pathlib import Path
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from app.state import AgentState
from app.logger_config import get_logger
from app.database import get_cumulative_context, save_interaction

# Chatterbox imports
from chatterbox.tts import ChatterboxTTS
import soundfile as sf

# Faster Whisper imports
from faster_whisper import WhisperModel

logger = get_logger(__name__)

# Initialize Local STT (Faster Whisper)
# Using 'base' model for speed/quality balance locally. 
# device="cpu" and compute_type="int8" for broad compatibility. 
# Use device="cuda" and compute_type="float16" if you have a GPU.
print("Loading Faster Whisper model...")
stt_model = WhisperModel("base", device="cpu", compute_type="int8")
print("Faster Whisper model loaded.")

# Initialize Local LLM (DeepSeek R1 via Ollama)
llm = ChatOllama(model="deepseek-r1:8b")

# Initialize Local TTS (Chatterbox)
# Note: This might take a moment to load the first time.
print("Loading Chatterbox TTS model...")
try:
    tts = ChatterboxTTS.from_pretrained(device="cpu")
except TypeError:
    # Fallback if signature is different (e.g. older versions)
    tts = ChatterboxTTS.from_pretrained()
print("Chatterbox TTS model loaded.")

def transcribe_audio(state: AgentState) -> AgentState:
    """Node to transcribe audio to text using Faster Whisper (Local)."""
    audio_path = state.get("input_audio_path")
    
    # If no audio provided, just return empty update (keeping existing input_text if any)
    if not audio_path or not os.path.exists(audio_path):
        return {}
    
    # Run transcription
    segments, info = stt_model.transcribe(audio_path, beam_size=5)
    
    # Combine segments into full text
    transcription_text = "".join([segment.text for segment in segments])
    
    return {"input_text": transcription_text}

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
        raise e
    
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
    # We store agent_thinking and cumilative_context (past) in state to save later
    return {
        "response_text": content, 
        "messages": [human_msg, response],
        "agent_thinking": agent_thinking,
        "cumilative_context": past_context
    }

def segment_text(state: AgentState) -> AgentState:
    """Node to split the agent's text response into smaller chunks for TTS."""
    text = state.get("response_text", "")
    
    if not text:
        return {"response_segments": []}
        
    # Split by common sentence terminators but keep them
    # distinct sentences usually end with . ? ! followed by space or newline
    # Simple regex split
    segments = re.split(r'(?<=[.!?])\s+', text)
    
    # Further splitting if still too long (heuristic limit ~200 chars)
    final_segments = []
    MAX_CHARS = 200
    
    for seg in segments:
        seg = seg.strip()
        if not seg:
            continue
            
        if len(seg) < MAX_CHARS:
            final_segments.append(seg)
        else:
            # Hard split on commas or just size if needed
            sub_parts = re.split(r'(?<=[,])\s+', seg)
            current_chunk = ""
            for part in sub_parts:
                if len(current_chunk) + len(part) < MAX_CHARS:
                    current_chunk += part + " "
                else:
                    if current_chunk:
                        final_segments.append(current_chunk.strip())
                    current_chunk = part + " "
            if current_chunk:
                final_segments.append(current_chunk.strip())
                
    logger.info(f"Segmented text into {len(final_segments)} chunks.")
    return {"response_segments": final_segments}

def refine_and_guardrail(state: AgentState) -> AgentState:
    """Node to validate and format chunks before synthesis."""
    segments = state.get("response_segments", [])
    refined_segments = []
    
    for seg in segments:
        # 1. Enforce UPPERCASE (as requested by user)
        seg = seg.upper()
        
        # 2. Check for completeness/formatting
        # (Simple heuristic: ensure it doesn't end strangely, though TTS handles most)
        
        # 3. Add padding for TTS stability (phantom space/char)
        # Chatterbox specific: padding helps avoid start/end truncation
        # We'll apply this in the synthesis step or here. Ideally here so it's "ready".
        # But actually, the prompt said "Padding your input text...". 
        # I will keep the raw text here pure for logging, and pad in synthesis.
        
        refined_segments.append(seg)
        
    logger.info("Segments refined and checked.")
    return {"response_segments": refined_segments}

def synthesize_audio(state: AgentState) -> AgentState:
    """Node to convert text segments to audio using Chatterbox TTS (Local) and concatenate them."""
    segments = state.get("response_segments", [])
    
    # Fallback to single text if segments are missing (backward compatibility)
    if not segments:
        text = state.get("response_text", "")
        if text:
            segments = [text]
        else:
             logger.warning("No text to synthesize.")
             return {"response_audio_path": None}
    
    logger.info(f"Synthesizing audio for {len(segments)} segments...")
    
    # Create dire
    output_dir = Path(__file__).parent.parent / "generated_audio"
    output_dir.mkdir(exist_ok=True)
    
    audio_arrays = []
    
    for i, seg in enumerate(segments):
        # Pad input to prevent truncation
        padded_seg = f" {seg} " 
        
        try:
            # Generate audio for the chunk
            # Retry logic could be added here if needed, but simple try/catch for now
            audio = tts.generate(padded_seg)
            
            if hasattr(audio, "numpy"):
                audio = audio.squeeze().numpy()
            elif hasattr(audio, "detach"): # torch tensor
                audio = audio.detach().cpu().squeeze().numpy()
                
            audio_arrays.append(audio)
            
            # Add silence between chunks (200ms)
            silence_samples = int(0.2 * tts.sr)
            silence = np.zeros(silence_samples, dtype=np.float32)
            audio_arrays.append(silence)
            
        except Exception as e:
            logger.error(f"TTS failed for segment '{seg}': {e}")
            # Continue to next segment? Or fail? 
            # Best to continue to partial result is better than nothing, 
            # but user might miss context. For now, log and continue.
            continue

    if not audio_arrays:
        logger.error("No audio generated.")
        return {"response_audio_path": None}
        
    # Concatenate all arrays
    final_audio = np.concatenate(audio_arrays)
    
    # Save final file
    filename = f"{uuid.uuid4()}.wav"
    speech_file_path = output_dir / filename
    
    try:
        sf.write(str(speech_file_path), final_audio, samplerate=tts.sr)
        logger.info(f"Final audio saved to {speech_file_path}")
    except Exception as e:
        logger.error(f"Failed to save audio file: {e}")
        raise e
    
    return {"response_audio_path": str(speech_file_path)}

def save_conversation(state: AgentState) -> AgentState:
    """Node to save the interaction to the database."""
    session_id = state.get("session_id")
    user_query = state.get("input_text")
    agent_answer = state.get("response_text")
    agent_thinking = state.get("agent_thinking", "")
    cumilative_context = state.get("cumilative_context", "") # content BEFORE this turn
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
