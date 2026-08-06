import sys
import os
import time

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
load_dotenv()

from edgevoice.orchestrator.executor import Executor

def test_pipeline():
    print("--- Starting Manual Verification ---")
    executor = Executor()
    
    # Test 1: Chat Intent
    try:
        print("\n[TEST 1] Chat Intent")
        res = executor.process_command("Hello, who are you?")
        print(f"Intent: {res['intent']}")
        print(f"Response: {res['response_text']}")
        print(f"Audio Path: {res['response_audio_path']}")
    except Exception as e:
        print(f"TEST 1 FAILED: {e}")
        
    # Test 2: Task Execution (File Creation)
    try:
        print("\n[TEST 2] Task Execution (File Creation)")
        test_file = "test_output.txt"
        if os.path.exists(test_file):
            os.remove(test_file)
            
        res = executor.process_command(f"Create a file named {test_file} with content 'Verified by PyVoiceAgent'")
        print(f"Intent: {res['intent']}")
        plan = res.get('plan')
        steps = plan.get('steps', []) if plan else []
        print(f"Plan Steps: {len(steps)}")
        print(f"Response: {res['response_text']}")
        
        if os.path.exists(test_file):
            print("SUCCESS: File created.")
            os.remove(test_file) # Cleanup
        else:
            print("FAILURE: File not created.")
    except Exception as e:
        print(f"TEST 2 FAILED: {e}")
            
    # Test 3: Memory Interaction
    try:
        print("\n[TEST 3] Memory Interaction")
        executor.process_command("My name is Abrar.")
        res = executor.process_command("What is my name?")
        print(f"Response: {res['response_text']}")
    except Exception as e:
        print(f"TEST 3 FAILED: {e}")

    # Test 4: Generation Audio Flag
    try:
        print("\n[TEST 4] Audio Generation Flag")
        res_text = executor.process_command("Hello", generate_audio=False)
        print(f"Text-to-Text Audio generated: {bool(res_text.get('response_audio_path'))}")

        res_voice = executor.process_command("Hello", generate_audio=True)
        print(f"Text-to-Voice Audio generated: {bool(res_voice.get('response_audio_path'))}")
        if res_voice.get('response_audio_path'):
             print(f"Audio path: {res_voice['response_audio_path']}")
    except Exception as e:
        print(f"TEST 4 FAILED: {e}")

if __name__ == "__main__":
    test_pipeline()
