from typing import Dict, Any, List
from app.core.llm import get_llm
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from app.schemas.plan_schema import ExecutionPlan, ToolCall

class PlannerAgent:
    def __init__(self):
        self.llm = get_llm(json_mode=True)
        
        self.parser = JsonOutputParser(pydantic_object=ExecutionPlan)
        
        self.prompt = PromptTemplate(
            template="""
            You are a Planner Agent. Your goal is to create a step-by-step execution plan to fulfill the user's request using the available tools.
            
            {tools_description}
            
            User Request: {input}
            Context/History: {context}
            
            Create a plan that breaks down the user's request into logical steps. Each step must use one of the available tools.
            If the request is simple, the plan may have only one step.
            
            Respond with a valid JSON object matching this schema:
            {format_instructions}
            """,
            input_variables=["input", "context", "tools_description"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )
        
        self.chain = self.prompt | self.llm | self.parser

    async def create_plan(self, user_input: str, context: str = "") -> ExecutionPlan:
        # Dynamically build tools description combining local tools and active MCP tools
        tools_desc = """
        AVAILABLE TOOLS:
        1. read_file(path: str): Reads content of a file.
        2. write_file(path: str, content: str): Writes content to a file (overwrites).
        3. list_directory(path: str): Lists files in a directory.
        4. run_python_script(script_content: str = None, script_path: str = None): Executes Python code.
        """
        
        try:
            from app.core.mcp_manager import MCPManager
            mcp_manager = MCPManager.get_instance()
            mcp_tools = await mcp_manager.list_tools()
            if mcp_tools:
                tools_desc += "\nMCP SERVER TOOLS:\n"
                for idx, t in enumerate(mcp_tools, start=5):
                    schema = t.get("input_schema", {})
                    props = schema.get("properties", {})
                    args_list = []
                    for k, v in props.items():
                        args_list.append(f"{k}: {v.get('type', 'string')}")
                    args_str = ", ".join(args_list)
                    tools_desc += f"{idx}. {t['name']}({args_str}): {t['description']}\n"
        except Exception as e:
            print(f"Error fetching dynamic MCP tools for planner: {e}")

        try:
            print(f"Creating plan for: {user_input}")
            result = await self.chain.ainvoke({
                "input": user_input, 
                "context": context,
                "tools_description": tools_desc
            })
            
            if not result or not isinstance(result, dict) or "steps" not in result:
                 raise ValueError(f"Invalid plan result (missing 'steps' key): {result}")
                 
            plan = ExecutionPlan(**result)
            print(f"Plan created with {len(plan.steps)} steps.")
            return plan
        except Exception as e:
            print(f"Planning failed: {e}")
            # Return empty plan or handle error
            return ExecutionPlan(goal=user_input, steps=[], estimated_complexity=0)

if __name__ == "__main__":
    agent = PlannerAgent()
    plan = agent.create_plan("Read requirements.txt and create a summary in summary.txt")
    print(plan.model_dump_json(indent=2))
