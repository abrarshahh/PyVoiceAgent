import sys
import os
import unittest
from unittest.mock import patch, MagicMock

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.core.permission_gate import request_permission
from app.tools.chrome_tool import ChromeAccessTool
from app.tools.gallery_tool import GalleryAccessTool
from app.tools.social_tool import SocialAccessTool
from app.orchestrator.executor import Executor
from app.schemas.plan_schema import ExecutionPlan, ToolCall

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

    @patch('webbrowser.open')
    def test_chrome_tool(self, mock_webbrowser):
        """Verify ChromeAccessTool opens search and URLs correctly."""
        tool = ChromeAccessTool()
        
        # Test URL opening
        res = tool.execute(url="https://google.com")
        self.assertTrue(res.success)
        mock_webbrowser.assert_called_with("https://google.com")
        
        # Test Search query opening
        res = tool.execute(search_query="Gemini AI")
        self.assertTrue(res.success)
        mock_webbrowser.assert_called_with("https://www.google.com/search?q=Gemini%20AI")

    def test_gallery_tool_list(self):
        """Verify GalleryAccessTool list action runs successfully."""
        tool = GalleryAccessTool()
        res = tool.execute(action="list")
        self.assertTrue(res.success)
        self.assertIn("images", res.output)
        self.assertIn("directory", res.output)

    @patch('webbrowser.open')
    def test_social_tool(self, mock_webbrowser):
        """Verify SocialAccessTool constructs correct social media URLs."""
        tool = SocialAccessTool()
        
        # Instagram username profile
        res = tool.execute(platform="instagram", username="cristiano")
        self.assertTrue(res.success)
        mock_webbrowser.assert_called_with("https://www.instagram.com/cristiano/")
        
        # Twitter search
        res = tool.execute(platform="twitter", search_query="ai news")
        self.assertTrue(res.success)
        mock_webbrowser.assert_called_with("https://x.com/search?q=ai%20news")

    @patch('app.core.permission_gate.request_permission', return_value=False)
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
        executor._planner = MagicMock()
        executor._planner.create_plan.return_value = mock_plan
        
        executor._intent_classifier = MagicMock()
        executor._intent_classifier.classify.return_value = MagicMock(intent="task_execution", confidence=1.0)
        
        executor._response_generator = MagicMock()
        executor._response_generator.generate_response.return_value = "Denied."
        
        executor._memory = MagicMock()
        executor._memory.retrieve_memory.return_value = []
        
        executor._tts = MagicMock()
        
        res = executor.process_command("Open Chrome to google.com", generate_audio=False)
        
        # Assert that permission request was triggered
        mock_request_permission.assert_called_once()
        
        # Verify execution log shows permission denied
        self.assertEqual(len(res["execution_log"]), 1)
        self.assertFalse(res["execution_log"][0]["success"])
        self.assertEqual(res["execution_log"][0]["error"], "Permission denied by user")

if __name__ == "__main__":
    unittest.main()
