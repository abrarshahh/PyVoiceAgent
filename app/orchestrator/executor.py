import time
import os
from typing import Dict, Any, List
from app.schemas.plan_schema import ExecutionState, ToolCall

class Executor:
    def __init__(self):
        print("Initializing Executor (Lazy Mode)...")
        self._stt = None
        self._intent_classifier = None
        self._planner = None
        self._response_generator = None
        self._tts = None
        self._memory = None
        self._tools = None

    @property
    def stt(self):
        if self._stt is None:
            from app.agents.stt_agent import STTAgent
            self._stt = STTAgent()
        return self._stt

    @property
    def intent_classifier(self):
        if self._intent_classifier is None:
            from app.agents.intent_agent import IntentAgent
            self._intent_classifier = IntentAgent()
        return self._intent_classifier

    @property
    def planner(self):
        if self._planner is None:
            from app.agents.planner_agent import PlannerAgent
            self._planner = PlannerAgent()
        return self._planner

    @property
    def response_generator(self):
        if self._response_generator is None:
            from app.agents.response_agent import ResponseAgent
            self._response_generator = ResponseAgent()
        return self._response_generator

    @property
    def tts(self):
        if self._tts is None:
            from app.tools.synthesizer import SynthesizerTool
            self._tts = SynthesizerTool()
        return self._tts

    @property
    def memory(self):
        if self._memory is None:
            from app.agents.memory_agent import MemoryAgent
            self._memory = MemoryAgent()
        return self._memory

    @property
    def tools(self):
        if self._tools is None:
            from app.tools.filesystem import ListDirTool, ReadFileTool, WriteFileTool
            from app.tools.python_runner import PythonRunnerTool
            self._tools = {
                "list_directory": ListDirTool(),
                "read_file": ReadFileTool(),
                "write_file": WriteFileTool(),
                "run_python_script": PythonRunnerTool()
            }
        return self._tools

    def process_command(self, text: str, generate_audio: bool = True) -> Dict[str, Any]:
        """
        Process a text command through the agentic pipeline.
        """
        results = {
            "transcription": text,
            "intent": None,
            "plan": None,
            "execution_log": [],
            "response_text": "",
            "response_audio_path": ""
        }
        
        if not text:
            return results
            
        print(f"Processing command: {text}")
        
        # 2. Memory Retrieval (Context)
        context = []
        if self.memory:
            context = self.memory.retrieve_memory(text)
        context_str = "\n".join(context)
        
        # 3. Intent Classification
        intent_res = self.intent_classifier.classify(text)
        results["intent"] = intent_res.intent
        
        if intent_res.intent == "task_execution":
            # 4. Planning
            plan = self.planner.create_plan(text, context=context_str)
            results["plan"] = plan.dict()
            
            # 5. Execution
            execution_log = []
            for step in plan.steps:
                tool_name = step.tool_name
                tool = self.tools.get(tool_name)
                
                if tool:
                    print(f"Executing {tool_name} with {step.arguments}...")
                    tool_res = tool.execute(**step.arguments)
                    execution_log.append({
                        "step_id": step.step_id,
                        "tool": tool_name,
                        "output": tool_res.output,
                        "error": tool_res.error,
                        "success": tool_res.success
                    })
                    
                    # Stop on critical failure (optional logic)
                    if not tool_res.success:
                        print(f"Step {step.step_id} failed: {tool_res.error}")
                        break
                else:
                     execution_log.append({
                        "step_id": step.step_id,
                        "tool": tool_name,
                        "error": "Tool not found",
                        "success": False
                    })
            
            results["execution_log"] = execution_log
            
            # 6. Response Generation
            response_text = self.response_generator.generate_response(text, execution_log)
            results["response_text"] = response_text
            
            # 7. Add to Memory
            if self.memory:
                self.memory.add_memory(f"User Request: {text}\nAction: Executed task\nResult: {response_text}")
                
        else:
            # Chat Mode
            response_text = self.response_generator.generate_response(text, [{"step_id": "0", "output": "Chat/Question answering mode"}])
            results["response_text"] = response_text
            
            if self.memory:
                 self.memory.add_memory(f"User Chat: {text}\nResponse: {response_text}")
 
        # 8. TTS (Conditional)
        if generate_audio:
            tts_res = self.tts.execute(text=response_text)
            if tts_res.success:
                results["response_audio_path"] = tts_res.output
            else:
                print(f"TTS failed: {tts_res.error}")
        
        return results

    def process_voice_command(self, audio_path: str) -> Dict[str, Any]:
        """
        Full pipeline: Audio -> Text -> [Process Command]
        """
        # 1. STT
        stt_result = self.stt.transcribe(audio_path)
        text = stt_result.get("text")
        
        if not text:
            return {
                "transcription": "",
                "intent": None,
                "plan": None,
                "execution_log": [],
                "response_text": "",
                "response_audio_path": ""
            }
            
        return self.process_command(text)

if __name__ == "__main__":
    # Test
    executor = Executor()
    # res = executor.process_voice_command("path/to/test.wav")
    # print(res)
