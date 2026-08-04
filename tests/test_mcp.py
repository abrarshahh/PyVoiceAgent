import sys
import os
import unittest
from unittest.mock import patch, MagicMock

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.core.mcp_manager import MCPManager
from app.orchestrator.executor import Executor
from app.schemas.plan_schema import ExecutionPlan, ToolCall

class TestMCPIntegration(unittest.IsolatedAsyncioTestCase):

    async def asyncSetUp(self):
        # Use get_instance and reset to ensure a fresh environment for testing
        self.mcp_manager = MCPManager.get_instance()
        # Reset mock pollution if any other test mocked the singleton's call_tool
        if hasattr(self.mcp_manager, "call_tool"):
            try:
                del self.mcp_manager.call_tool
            except AttributeError:
                pass
        # Reset the internal states
        self.mcp_manager._initialized = False
        self.mcp_manager._sessions.clear()
        self.mcp_manager._contexts.clear()

    async def asyncTearDown(self):
        # Shut down any subprocesses started during the tests
        await self.mcp_manager.shutdown()

    async def test_mcp_server_initialization_and_tool_list(self):
        """Verify that the MCP manager starts the custom server and lists its tools."""
        await self.mcp_manager.initialize()
        
        # Verify system_apps is in sessions
        self.assertIn("system_apps", self.mcp_manager._sessions)
        
        tools = await self.mcp_manager.list_tools()
        self.assertTrue(len(tools) >= 3, f"Expected at least 3 tools from system_apps server, got {len(tools)}")
        
        tool_names = [t["name"] for t in tools]
        self.assertIn("chrome_access", tool_names)
        self.assertIn("gallery_access", tool_names)
        self.assertIn("social_access", tool_names)

    async def test_mcp_tool_execution(self):
        """Verify calling an MCP tool routes correctly and returns tool output."""
        await self.mcp_manager.initialize()
        
        # Test call_tool on system_apps for gallery_access (no browser launch side effect)
        res = await self.mcp_manager.call_tool(
            server_name="system_apps",
            tool_name="gallery_access",
            arguments={"action": "list"}
        )
        self.assertIn("Directory:", res)
        self.assertIn("Images:", res)

    @patch('app.core.permission_gate.request_permission', return_value=True)
    async def test_executor_with_mcp_success(self, mock_request_permission):
        """Verify the Executor routes tool execution to MCP server when permission is granted."""
        executor = Executor()
        
        mock_plan = ExecutionPlan(
            goal="Open a website",
            steps=[
                ToolCall(tool_name="chrome_access", arguments={"url": "https://mcp.dev"}, step_id="1", reasoning="to verify mcp integration")
            ],
            estimated_complexity=1
        )
        
        # Inject mocks to avoid real LLM calls
        from unittest.mock import AsyncMock
        executor._planner = MagicMock()
        executor._planner.create_plan = AsyncMock(return_value=mock_plan)
        executor._intent_classifier = MagicMock()
        executor._intent_classifier.classify.return_value = MagicMock(intent="task_execution", confidence=1.0)
        executor._response_generator = MagicMock()
        executor._response_generator.generate_response.return_value = "Successfully completed MCP task."
        executor._memory = MagicMock()
        executor._memory.retrieve_memory.return_value = []
        executor._tts = MagicMock()
        
        # Mock mcp_manager call_tool directly on the executor's instance
        executor.mcp_manager.call_tool = AsyncMock(return_value="Opened URL https://mcp.dev")
        
        res = await executor.process_command("Open website", generate_audio=False)
        
        # Assert permission gate was checked
        mock_request_permission.assert_called_once_with("chrome_access", {"url": "https://mcp.dev"})
        
        # Assert call_tool was routed to MCP manager
        executor.mcp_manager.call_tool.assert_called_once_with(
            "system_apps", "chrome_access", {"url": "https://mcp.dev"}
        )
        
        # Assert execution log is success
        self.assertEqual(len(res["execution_log"]), 1)
        self.assertTrue(res["execution_log"][0]["success"])
        self.assertEqual(res["execution_log"][0]["output"], "Opened URL https://mcp.dev")

if __name__ == "__main__":
    unittest.main()
