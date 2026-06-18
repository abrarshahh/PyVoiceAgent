from chatterbox.tts import ChatterboxTTS
import os
import sys

print("--- TTS Diagnostic ---")
try:
    print("Initializing ChatterboxTTS...")
    tts = ChatterboxTTS.from_pretrained(device="cpu")
    print("Initialization successful.")
    
    text = "Hello, this is a test."
    print(f"Generating audio for: {text}")
    res = tts.generate(text)
    
    print(f"Result type: {type(res)}")
    print(f"Result content: {res}")
    
    if isinstance(res, tuple):
        print(f"Tuple length: {len(res)}")
        for i, item in enumerate(res):
            print(f"  Item {i} type: {type(item)}")
            
except Exception as e:
    print(f"CRITICAL ERROR: {e}")
    import traceback
    traceback.print_exc()
