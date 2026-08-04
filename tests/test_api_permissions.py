import sys
import os
import unittest
from unittest.mock import patch, MagicMock, AsyncMock
from fastapi.testclient import TestClient

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.main import app
from app.api.routes import executor
from app.schemas.plan_schema import ExecutionPlan, ToolCall

class TestAPIPermissions(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        
        # Reset permission_manager pending tasks
        from app.core import permission_manager
        permission_manager._pending_tasks.clear()

    @patch('app.api.routes.executor.process_command')
    def test_text_to_text_chat_intent(self, mock_process_command):
        """Verify chat intent returns immediately."""
        # Create an async mock helper since process_command is async
        async def mock_async_process(*args, **kwargs):
            return {
                "transcription": "Hello",
                "intent": "chat",
                "plan": None,
                "response_text": "Hello there!",
                "status": "completed"
            }
        mock_process_command.side_effect = mock_async_process
        
        response = self.client.post("/text-to-text", json={"text": "Hello", "session_id": "test-session"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "completed")
        self.assertEqual(data["response_text"], "Hello there!")

    @patch('app.api.routes.executor.process_command')
    def test_text_to_text_task_intent_gating(self, mock_process_command):
        """Verify task execution intent is gated and saved as pending."""
        mock_plan = {
            "goal": "open browser",
            "steps": [{"tool_name": "chrome_access", "arguments": {"url": "https://test.com"}, "step_id": "1", "reasoning": "open website"}],
            "estimated_complexity": 1
        }
        
        async def mock_async_process(*args, **kwargs):
            return {
                "transcription": "open browser",
                "intent": "task_execution",
                "plan": mock_plan,
                "past_context": "dummy context",
                "status": "pending_permission"
            }
        mock_process_command.side_effect = mock_async_process
        
        response = self.client.post("/text-to-text", json={"text": "open browser", "session_id": "session-123"})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        # Check that we received pending status
        self.assertEqual(data["status"], "pending_permission")
        self.assertEqual(data["session_id"], "session-123")
        self.assertIn("plan", data)
        
        # Check that it is saved in permission_manager
        from app.core import permission_manager
        pending_task = permission_manager.get_pending_task("session-123")
        self.assertIsNotNone(pending_task)
        self.assertEqual(pending_task["original_text"], "open browser")

    @patch('app.api.routes.executor.execute_plan')
    def test_permissions_respond_approval(self, mock_execute_plan):
        """Verify that approving a pending task executes the plan."""
        from app.core import permission_manager
        
        # Manually save a pending task
        permission_manager.add_pending_task(
            session_id="session-456",
            plan={"goal": "test", "steps": []},
            original_text="open browser",
            generate_audio=False,
            past_context="context"
        )
        
        async def mock_async_execute(*args, **kwargs):
            return {
                "transcription": "open browser",
                "intent": "task_execution",
                "execution_log": [{"step_id": "1", "tool": "chrome_access", "success": True}],
                "response_text": "Chrome opened.",
                "status": "completed"
            }
        mock_execute_plan.side_effect = mock_async_execute
        
        # Approve task
        response = self.client.post("/permissions/respond", json={"session_id": "session-456", "approved": True})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        self.assertEqual(data["status"], "completed")
        self.assertEqual(data["response_text"], "Chrome opened.")
        
        # Assert plan execution was triggered
        mock_execute_plan.assert_called_once()
        
        # Assert task was cleared
        self.assertIsNone(permission_manager.get_pending_task("session-456"))

    def test_permissions_respond_rejection(self):
        """Verify that rejecting a pending task clears it without executing."""
        from app.core import permission_manager
        permission_manager.add_pending_task(
            session_id="session-789",
            plan={"goal": "test", "steps": []},
            original_text="open browser",
            generate_audio=False,
            past_context="context"
        )
        
        # Reject task
        response = self.client.post("/permissions/respond", json={"session_id": "session-789", "approved": False})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        
        self.assertEqual(data["status"], "rejected")
        
        # Assert task was cleared
        self.assertIsNone(permission_manager.get_pending_task("session-789"))

    def test_get_mcp_servers(self):
        """Verify mcp servers lists configured servers."""
        response = self.client.get("/mcp/servers")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("servers", data)
        self.assertIn("system_apps", data["servers"])

if __name__ == "__main__":
    unittest.main()
