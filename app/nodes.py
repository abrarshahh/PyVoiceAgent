import os
import uuid
from pathlib import Path
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from app.state import AgentState

# Chatterbox imports
from chatterbox.tts import ChatterboxTTS
import soundfile as sf

# Faster Whisper imports
from faster_whisper import WhisperModel

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

import re
import emoji
from app.logger_config import get_logger

logger = get_logger(__name__)

def process_input(state: AgentState) -> AgentState:
    """Node to process text input and generate a textual response."""
    text = state.get("input_text", "")
    messages = state.get("messages", [])
    
    logger.info("Processing input text.")
    logger.agent_output(f"User Input: {text}")
    
    # Simple prompt for the agent
    system_message = SystemMessage(content="You are a helpful voice assistant. Keep your responses concise and conversational.")
    
    # If using add_messages, we don't need to manually manage the whole list in the invocation
    # But usually for the LLM invoke, we pass everything.
    # If the state already has messages, use them. If empty (new thread), add system.
    
    if not messages:
        messages = [system_message]
    
    # We don't need to append HumanMessage here if it was already added by a previous node? 
    # Actually, in our graph, no one added the HumanMessage yet.  
    # Ideally, we should add it to the state.
    
    # Let's construct the prompt messages
    prompt_messages = list(messages)
    # If the last message is NOT the current input(text), we append it for the LLM
    # In `chat_text`, we put `input_text` in state, but didn't add to `messages`.
    # So we construct the prompt here.
    
    # Note: With add_messages, we return the NEW messages to be added.
    
    human_msg = HumanMessage(content=text)
    prompt_messages.append(human_msg)
    
    # Invoke the local LLM
    try:
        response = llm.invoke(prompt_messages)
    except Exception as e:
        logger.error(f"LLM invocation failed: {e}")
        raise e
    
    # Filter out <think>...</think> tags from DeepSeek R1
    content = response.content
    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
    
    # Remove emojis
    content = emoji.replace_emoji(content, replace='')
    
    # Update the response content with cleaned text
    response.content = content
    
    logger.agent_output(f"Agent Response: {content}")
    
    # Return the new messages to be appended to history
    return {"response_text": content, "messages": [human_msg, response]}

def synthesize_audio(state: AgentState) -> AgentState:
    """Node to convert text response to audio using Chatterbox TTS (Local)."""
    text = state.get("response_text", "")
    
    if not text:
        logger.warning("No text to synthesize.")
        return {"response_audio_path": None}
    
    logger.info("Synthesizing audio...")
    
    # Create directory if it doesn't exist
    output_dir = Path(__file__).parent.parent / "generated_audio"
    output_dir.mkdir(exist_ok=True)
    
    # Save as .wav for Chatterbox
    filename = f"{uuid.uuid4()}.wav"
    speech_file_path = output_dir / filename
    
    # Generate audio
    try:
        audio = tts.generate(text)
        
        # Convert from Tensor to Numpy if needed
        if hasattr(audio, "numpy"):
            audio = audio.squeeze().numpy()
        
        # Save to file
        sf.write(str(speech_file_path), audio, samplerate=tts.sr)
        logger.info(f"Audio saved to {speech_file_path}")
        
    except Exception as e:
        logger.error(f"TTS generation failed: {e}")
        raise e
    
    return {"response_audio_path": str(speech_file_path)}
