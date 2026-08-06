"""The single async pipeline for EdgeVoice.

This is the only orchestration entry point. LangGraph is gone.
The pipeline is: vad -> stt -> memory_recall -> intent -> plan ->
policy -> execute -> respond -> tts -> audit.

Voice nodes (vad, stt, tts) and audit land in later phases. For now
the executor handles text input, intent classification, planning,
tool execution, and TTS synthesis via the existing tools.
"""
import time
from typing import Dict, Any, List

from edgevoice.agents.assistant import generate_response
from edgevoice.orchestrator.router import IntentAgent
from edgevoice.orchestrator.planner import PlannerAgent
from edgevoice.core.memory import MemoryAgent
from edgevoice.core.mcp import MCPManager
from edgevoice.core.permission_gate import request_permission
from edgevoice.schemas.plan import ExecutionPlan
from edgevoice.core.stt import transcribe_audio
from edgevoice.core.tts import SynthesizerTool
from edgevoice.tools.filesystem_legacy import ListDirTool, ReadFileTool, WriteFileTool
from edgevoice.core.audit import save_conversation


class Executor:
    def __init__(self):
        print("Initializing Executor (v2 single-pipeline)...")
        # Eagerly build the small, dependency-free pieces; lazy-build the heavy ones.
        self.intent_classifier = IntentAgent()
        self.planner = PlannerAgent()
        self.memory = MemoryAgent()
        self.tts = SynthesizerTool()
        self.mcp_manager = MCPManager.get_instance()

        # Local tool registry. run_python_script was removed in Phase 0;
        # it will be replaced by the sandboxed mcp-shell skill in Phase 3.
        self.tools = {
            "list_directory": ListDirTool(),
            "read_file": ReadFileTool(),
            "write_file": WriteFileTool(),
        }

    async def process_command(
        self,
        text: str,
        generate_audio: bool = True,
        plan_only: bool = False,
        session_id: str | None = None,
    ) -> Dict[str, Any]:
        """Process a text command through the agentic pipeline."""
        results: Dict[str, Any] = {
            "transcription": text,
            "intent": None,
            "plan": None,
            "execution_log": [],
            "response_text": "",
            "response_audio_path": "",
            "status": "completed",
        }

        if not text:
            return results

        print(f"Processing command: {text}")

        # 1. Memory Retrieval (Context)
        context: List[str] = []
        try:
            context = self.memory.retrieve_memory(text) or []
        except Exception as e:
            print(f"Memory retrieval failed: {e}")
        context_str = "\n".join(context)

        # 2. Intent Classification
        intent_res = self.intent_classifier.classify(text)
        results["intent"] = intent_res.intent

        if intent_res.intent == "task_execution":
            # 3. Planning
            plan = await self.planner.create_plan(text, context=context_str)
            results["plan"] = (
                plan.model_dump() if hasattr(plan, "model_dump") else plan.dict()
            )

            if plan_only:
                results["status"] = "pending_permission"
                results["past_context"] = context_str
                return results

            # 4. Execution
            execution_log: List[Dict[str, Any]] = []
            for step in plan.steps:
                tool_name = step.tool_name

                # Check MCP tools first
                mcp_tool_info = None
                try:
                    mcp_tools = await self.mcp_manager.list_tools()
                    mcp_tool_info = next(
                        (t for t in mcp_tools if t["name"] == tool_name), None
                    )
                except Exception as e:
                    print(f"Error checking MCP tools: {e}")

                if mcp_tool_info:
                    server_name = mcp_tool_info["server_name"]
                    approved = request_permission(tool_name, step.arguments)
                    if not approved:
                        print(
                            f"Execution of step {step.step_id} ({tool_name}) on "
                            f"MCP server '{server_name}' was denied by user."
                        )
                        execution_log.append(
                            {
                                "step_id": step.step_id,
                                "tool": tool_name,
                                "output": None,
                                "error": "Permission denied by user",
                                "success": False,
                            }
                        )
                        break

                    print(
                        f"Executing MCP tool '{tool_name}' on server "
                        f"'{server_name}' with {step.arguments}..."
                    )
                    try:
                        output = await self.mcp_manager.call_tool(
                            server_name, tool_name, step.arguments
                        )
                        execution_log.append(
                            {
                                "step_id": step.step_id,
                                "tool": tool_name,
                                "output": output,
                                "error": None,
                                "success": True,
                            }
                        )
                    except Exception as e:
                        execution_log.append(
                            {
                                "step_id": step.step_id,
                                "tool": tool_name,
                                "output": None,
                                "error": str(e),
                                "success": False,
                            }
                        )
                        break
                else:
                    tool = self.tools.get(tool_name)
                    if tool:
                        approved = request_permission(tool_name, step.arguments)
                        if not approved:
                            print(
                                f"Execution of step {step.step_id} ({tool_name}) "
                                f"was denied by user."
                            )
                            execution_log.append(
                                {
                                    "step_id": step.step_id,
                                    "tool": tool_name,
                                    "output": None,
                                    "error": "Permission denied by user",
                                    "success": False,
                                }
                            )
                            break

                        print(f"Executing local tool '{tool_name}' with {step.arguments}...")
                        tool_res = tool.execute(**step.arguments)
                        execution_log.append(
                            {
                                "step_id": step.step_id,
                                "tool": tool_name,
                                "output": tool_res.output,
                                "error": tool_res.error,
                                "success": tool_res.success,
                            }
                        )
                        if not tool_res.success:
                            print(f"Step {step.step_id} failed: {tool_res.error}")
                            break
                    else:
                        execution_log.append(
                            {
                                "step_id": step.step_id,
                                "tool": tool_name,
                                "error": (
                                    f"Tool '{tool_name}' not found locally or on "
                                    f"any MCP server"
                                ),
                                "success": False,
                            }
                        )
                        break

            results["execution_log"] = execution_log

            # 5. Response Generation
            response = generate_response(
                text,
                session_id=session_id,
                execution_log=execution_log,
                execution_mode="task",
            )
            results["response_text"] = response["response_text"]

            # 6. Add to Memory
            try:
                self.memory.add_memory(
                    f"User Request: {text}\nAction: Executed task\n"
                    f"Result: {response['response_text']}"
                )
            except Exception as e:
                print(f"Memory write failed: {e}")
        else:
            # Chat Mode
            response = generate_response(
                text, session_id=session_id, execution_mode="chat"
            )
            results["response_text"] = response["response_text"]
            try:
                self.memory.add_memory(
                    f"User Chat: {text}\nResponse: {response['response_text']}"
                )
            except Exception as e:
                print(f"Memory write failed: {e}")

        # 7. TTS (Conditional)
        if generate_audio:
            tts_res = self.tts.execute(text=results["response_text"])
            if tts_res.success:
                results["response_audio_path"] = tts_res.output
            else:
                print(f"TTS failed: {tts_res.error}")

        # 8. Persistence
        if session_id and results["response_text"]:
            try:
                save_conversation(
                    session_id=session_id,
                    user_query=text,
                    agent_answer=results["response_text"],
                )
            except Exception as e:
                print(f"Persistence failed: {e}")

        return results

    async def execute_plan(
        self,
        plan: Any,
        text: str,
        generate_audio: bool = True,
        past_context: str = "",
        session_id: str | None = None,
    ) -> Dict[str, Any]:
        """Execute a pre-approved plan without terminal permission prompts."""
        if isinstance(plan, dict):
            plan = ExecutionPlan(**plan)

        print(f"Executing pre-approved plan for task: {text}")

        execution_log: List[Dict[str, Any]] = []
        for step in plan.steps:
            tool_name = step.tool_name

            mcp_tool_info = None
            try:
                mcp_tools = await self.mcp_manager.list_tools()
                mcp_tool_info = next(
                    (t for t in mcp_tools if t["name"] == tool_name), None
                )
            except Exception as e:
                print(f"Error checking MCP tools: {e}")

            if mcp_tool_info:
                server_name = mcp_tool_info["server_name"]
                print(
                    f"Executing MCP tool '{tool_name}' on server "
                    f"'{server_name}' with {step.arguments}..."
                )
                try:
                    output = await self.mcp_manager.call_tool(
                        server_name, tool_name, step.arguments
                    )
                    execution_log.append(
                        {
                            "step_id": step.step_id,
                            "tool": tool_name,
                            "output": output,
                            "error": None,
                            "success": True,
                        }
                    )
                except Exception as e:
                    execution_log.append(
                        {
                            "step_id": step.step_id,
                            "tool": tool_name,
                            "output": None,
                            "error": str(e),
                            "success": False,
                        }
                    )
                    break
            else:
                tool = self.tools.get(tool_name)
                if tool:
                    print(f"Executing local tool '{tool_name}' with {step.arguments}...")
                    tool_res = tool.execute(**step.arguments)
                    execution_log.append(
                        {
                            "step_id": step.step_id,
                            "tool": tool_name,
                            "output": tool_res.output,
                            "error": tool_res.error,
                            "success": tool_res.success,
                        }
                    )
                    if not tool_res.success:
                        print(f"Step {step.step_id} failed: {tool_res.error}")
                        break
                else:
                    execution_log.append(
                        {
                            "step_id": step.step_id,
                            "tool": tool_name,
                            "error": (
                                f"Tool '{tool_name}' not found locally or on "
                                f"any MCP server"
                            ),
                            "success": False,
                        }
                    )
                    break

        response = generate_response(
            text,
            session_id=session_id,
            execution_log=execution_log,
            execution_mode="task",
        )

        try:
            self.memory.add_memory(
                f"User Request: {text}\nAction: Executed task\n"
                f"Result: {response['response_text']}"
            )
        except Exception as e:
            print(f"Memory write failed: {e}")

        response_audio_path = ""
        if generate_audio:
            tts_res = self.tts.execute(text=response["response_text"])
            if tts_res.success:
                response_audio_path = tts_res.output
            else:
                print(f"TTS failed: {tts_res.error}")

        if session_id and response["response_text"]:
            try:
                save_conversation(
                    session_id=session_id,
                    user_query=text,
                    agent_answer=response["response_text"],
                )
            except Exception as e:
                print(f"Persistence failed: {e}")

        return {
            "transcription": text,
            "intent": "task_execution",
            "plan": plan.model_dump() if hasattr(plan, "model_dump") else plan.dict(),
            "execution_log": execution_log,
            "response_text": response["response_text"],
            "response_audio_path": response_audio_path,
            "status": "completed",
        }

    async def process_voice_command(self, audio_path: str) -> Dict[str, Any]:
        """Audio in -> Text in -> Process."""
        text = transcribe_audio(audio_path)
        if not text:
            return {
                "transcription": "",
                "intent": None,
                "plan": None,
                "execution_log": [],
                "response_text": "",
                "response_audio_path": "",
            }
        return await self.process_command(text)


if __name__ == "__main__":
    import asyncio

    executor = Executor()
    print("Executor ready. Call `await executor.process_command('hello')` to run.")
