import sys
import os
import unittest
from unittest.mock import patch, MagicMock

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from edgevoice.core.permission_gate import request_permission
from edgevoice.tools.chrome_legacy import ChromeAccessTool
from edgevoice.tools.gallery_legacy import GalleryAccessTool
from edgevoice.tools.social_legacy import SocialAccessTool
from edgevoice.orchestrator.executor import Executor
from edgevoice.schemas.plan import ExecutionPlan, ToolCall

class TestPermissionGateAndTools(unittest.TestCase):

    @patch('sys.stdin.readline', return_value='y\n')
    def test_permission_gate_granted(self, mock_readline):
        """Verify that typing 'y' grants permission."""
        res = request_permission("dummy_tool", {"arg1": "val1"})
        self.assertTrue(res)

    @patch('sys.stdin.readline', return_value='n\n')
    def test_permission_gate_denied(self, mock_readline):
        """Verify that typing 'n' denies permission."""
        res = request_permission("dummy_tool", {"arg1": "val1"})
        self.assertFalse(res)

    @patch('sys.stdin.readline', return_value='\n')
    def test_permission_gate_default_denied(self, mock_readline):
        """Verify that typing empty string denies permission."""
        res = request_permission("dummy_tool", {})
        self.assertFalse(res)

    def test_chrome_tool_deprecated(self):
        """Verify ChromeAccessTool raises NotImplementedError."""
        tool = ChromeAccessTool()
        with self.assertRaises(NotImplementedError):
            tool.execute(url="https://google.com")

    def test_gallery_tool_deprecated(self):
        """Verify GalleryAccessTool raises NotImplementedError."""
        tool = GalleryAccessTool()
        with self.assertRaises(NotImplementedError):
            tool.execute(action="list")

    def test_social_tool_deprecated(self):
        """Verify SocialAccessTool raises NotImplementedError."""
        tool = SocialAccessTool()
        with self.assertRaises(NotImplementedError):
            tool.execute(platform="instagram", username="cristiano")

    @patch('edgevoice.core.permission_gate.request_permission', return_value=False)
    def test_executor_gating_denied(self, mock_request_permission):
        """Verify that when permission is denied, the executor aborts execution."""
        executor = Executor()
        
        # Mock planner to return a task execution plan without instantiating real agent
        mock_plan = ExecutionPlan(
            goal="Open Chrome to google.com",
            steps=[
                ToolCall(tool_name="chrome_access", arguments={"url": "https://google.com"}, step_id="1", reasoning="to open google")
            ],
            estimated_complexity=1
        )
        
        # Inject mocks directly into the private fields of Executor to bypass real agent loading
        executor.planner = MagicMock()
        executor.planner.create_plan.return_value = mock_plan
        
        executor.intent_classifier = MagicMock()
        executor.intent_classifier.classify.return_value = MagicMock(intent="task_execution", confidence=1.0)
        
        # Mock generate_response import from edgevoice.agents.assistant
        with patch('edgevoice.orchestrator.executor.generate_response', return_value={"response_text": "Denied."}):
            res = executor.process_command("Open Chrome to google.com", generate_audio=False)
            
            # Assert that permission request was triggered
            mock_request_permission.assert_called_once()
            
            # Verify execution log shows permission denied
            self.assertEqual(len(res["execution_log"]), 1)
            self.assertFalse(res["execution_log"][0]["success"])
            self.assertEqual(res["execution_log"][0]["error"], "Permission denied by user")

if __name__ == "__main__":
    unittest.main()
