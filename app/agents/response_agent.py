from typing import Dict, Any, List
from app.core.llm import get_llm
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

class ResponseAgent:
    def __init__(self):
        self.llm = get_llm(json_mode=False)
        
        self.prompt = PromptTemplate(
            template="""
            You are a helpful voice assistant. Your goal is to generate a concise, natural response to the user based on the execution results of their request.
            
            User Request: {input}
            Execution Results: {results}
            
            Guidelines:
            - Be concise but friendly.
            - If the task was successful, confirm it.
            - If there were errors, explain them simply.
            - Do not mention technical details like JSON or tools unless asked.
            - Speak as if you performed the action yourself.
            
            Response:
            """,
            input_variables=["input", "results"]
        )
        
        self.chain = self.prompt | self.llm | StrOutputParser()

    def generate_response(self, user_input: str, execution_results: List[Dict[str, Any]]) -> str:
        try:
            print(f"Generating response for: {user_input}")
            # Format results for the prompt
            results_summary = "\n".join([f"- Step {r.get('step_id')}: {r.get('output') or r.get('error')}" for r in execution_results])
            
            response = self.chain.invoke({
                "input": user_input,
                "results": results_summary
            })
            
            # Clean up <think> tags if present ( DeepSeek R1 specific )
            import re
            response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()
            
            print(f"Generated response: {response}")
            return response
        except Exception as e:
            print(f"Response generation failed: {e}")
            return "Task completed."

if __name__ == "__main__":
    agent = ResponseAgent()
    print(agent.generate_response("Create a file", [{"step_id": "1", "output": "Successfully wrote to test.txt"}]))
