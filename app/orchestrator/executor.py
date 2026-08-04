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
        self._mcp_manager = None

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
    def mcp_manager(self):
        if self._mcp_manager is None:
            from app.core.mcp_manager import MCPManager
            self._mcp_manager = MCPManager.get_instance()
        return self._mcp_manager

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

    async def process_command(self, text: str, generate_audio: bool = True, plan_only: bool = False) -> Dict[str, Any]:
        """
        Process a text command through the agentic pipeline.
        """
        results = {
            "transcription": text,
            "intent": None,
            "plan": None,
            "execution_log": [],
            "response_text": "",
            "response_audio_path": "",
            "status": "completed"
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
            plan = await self.planner.create_plan(text, context=context_str)
            results["plan"] = plan.model_dump() if hasattr(plan, "model_dump") else plan.dict()
            
            if plan_only:
                results["status"] = "pending_permission"
                results["past_context"] = context_str
                return results
            
            # 5. Execution
            execution_log = []
            for step in plan.steps:
                tool_name = step.tool_name
                
                # Check MCP tools first
                try:
                    mcp_tools = await self.mcp_manager.list_tools()
                    mcp_tool_info = next((t for t in mcp_tools if t["name"] == tool_name), None)
                except Exception as e:
                    print(f"Error checking MCP tools: {e}")
                    mcp_tool_info = None
                
                if mcp_tool_info:
                    server_name = mcp_tool_info["server_name"]
                    from app.core.permission_gate import request_permission
                    approved = request_permission(tool_name, step.arguments)
                    
                    if not approved:
                        print(f"Execution of step {step.step_id} ({tool_name}) on MCP server '{server_name}' was denied by user.")
                        execution_log.append({
                            "step_id": step.step_id,
                            "tool": tool_name,
                            "output": None,
                            "error": "Permission denied by user",
                            "success": False
                        })
                        break
                    
                    print(f"Executing MCP tool '{tool_name}' on server '{server_name}' with {step.arguments}...")
                    try:
                        output = await self.mcp_manager.call_tool(server_name, tool_name, step.arguments)
                        execution_log.append({
                            "step_id": step.step_id,
                            "tool": tool_name,
                            "output": output,
                            "error": None,
                            "success": True
                        })
                    except Exception as e:
                        execution_log.append({
                            "step_id": step.step_id,
                            "tool": tool_name,
                            "output": None,
                            "error": str(e),
                            "success": False
                        })
                        break
                else:
                    # Fallback to local tool
                    tool = self.tools.get(tool_name)
                    if tool:
                        from app.core.permission_gate import request_permission
                        approved = request_permission(tool_name, step.arguments)
                        
                        if not approved:
                            print(f"Execution of step {step.step_id} ({tool_name}) was denied by user.")
                            execution_log.append({
                                "step_id": step.step_id,
                                "tool": tool_name,
                                "output": None,
                                "error": "Permission denied by user",
                                "success": False
                            })
                            break
                        
                        print(f"Executing local tool '{tool_name}' with {step.arguments}...")
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
                            "error": f"Tool '{tool_name}' not found locally or on any MCP server",
                            "success": False
                        })
                        break
            
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

    async def execute_plan(self, plan: Any, text: str, generate_audio: bool = True, past_context: str = "") -> Dict[str, Any]:
        """
        Execute a pre-approved plan without gating it with a terminal permission check.
        """
        from app.schemas.plan_schema import ExecutionPlan
        
        # Convert dict to ExecutionPlan if necessary
        if isinstance(plan, dict):
            plan = ExecutionPlan(**plan)
            
        print(f"Executing pre-approved plan for task: {text}")
        
        execution_log = []
        for step in plan.steps:
            tool_name = step.tool_name
            
            # Check MCP tools first
            try:
                mcp_tools = await self.mcp_manager.list_tools()
                mcp_tool_info = next((t for t in mcp_tools if t["name"] == tool_name), None)
            except Exception as e:
                print(f"Error checking MCP tools: {e}")
                mcp_tool_info = None
            
            if mcp_tool_info:
                server_name = mcp_tool_info["server_name"]
                print(f"Executing MCP tool '{tool_name}' on server '{server_name}' with {step.arguments}...")
                try:
                    output = await self.mcp_manager.call_tool(server_name, tool_name, step.arguments)
                    execution_log.append({
                        "step_id": step.step_id,
                        "tool": tool_name,
                        "output": output,
                        "error": None,
                        "success": True
                    })
                except Exception as e:
                    execution_log.append({
                        "step_id": step.step_id,
                        "tool": tool_name,
                        "output": None,
                        "error": str(e),
                        "success": False
                    })
                    break
            else:
                # Fallback to local tool
                tool = self.tools.get(tool_name)
                if tool:
                    print(f"Executing local tool '{tool_name}' with {step.arguments}...")
                    tool_res = tool.execute(**step.arguments)
                    execution_log.append({
                        "step_id": step.step_id,
                        "tool": tool_name,
                        "output": tool_res.output,
                        "error": tool_res.error,
                        "success": tool_res.success
                    })
                    
                    if not tool_res.success:
                        print(f"Step {step.step_id} failed: {tool_res.error}")
                        break
                else:
                    execution_log.append({
                        "step_id": step.step_id,
                        "tool": tool_name,
                        "error": f"Tool '{tool_name}' not found locally or on any MCP server",
                        "success": False
                    })
                    break
                    
        # 6. Response Generation
        response_text = self.response_generator.generate_response(text, execution_log)
        
        # 7. Add to Memory
        if self.memory:
            self.memory.add_memory(f"User Request: {text}\nAction: Executed task\nResult: {response_text}")
            
        # 8. TTS (Conditional)
        response_audio_path = ""
        if generate_audio:
            tts_res = self.tts.execute(text=response_text)
            if tts_res.success:
                response_audio_path = tts_res.output
            else:
                print(f"TTS failed: {tts_res.error}")
                
        return {
            "transcription": text,
            "intent": "task_execution",
            "plan": plan.model_dump() if hasattr(plan, "model_dump") else plan.dict(),
            "execution_log": execution_log,
            "response_text": response_text,
            "response_audio_path": response_audio_path,
            "status": "completed"
        }

    async def process_voice_command(self, audio_path: str) -> Dict[str, Any]:
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
            
        return await self.process_command(text)

if __name__ == "__main__":
    # Test
    import asyncio
    executor = Executor()
    # res = asyncio.run(executor.process_voice_command("path/to/test.wav"))
    # print(res)
