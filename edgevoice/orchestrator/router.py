from typing import Dict, Any
from edgevoice.core.llm import get_llm
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from edgevoice.schemas.plan import IntentClassification

class IntentAgent:
    def __init__(self):
        try:
            self.llm = get_llm(json_mode=True)
            self.parser = JsonOutputParser(pydantic_object=IntentClassification)
            
            self.prompt = PromptTemplate(
                template="""
                You are a highly accurate intent classification agent.
                
                Classification categories:
                - task_execution: Requests to create/read/list files, run code, or perform system actions.
                - chat: Greetings, casual conversation, questions about yourself, or general knowledge.
                
                Return ONLY the JSON. No preamble, no explanation.
                Example: {{"intent": "task_execution", "task_type": "file_management", "confidence": 0.98}}
                
                User Input: {input}
                """,
                input_variables=["input"],
                partial_variables={"format_instructions": self.parser.get_format_instructions()}
            )
            self.chain = self.prompt | self.llm | self.parser
        except NotImplementedError:
            self.llm = None
            self.chain = None

    def classify(self, text: str) -> IntentClassification:
        if self.chain is None:
            print("LLM is not configured. Defaulting intent to 'chat'.")
            return IntentClassification(intent="chat", confidence=1.0)
        try:
            print(f"Classifying intent for: {text}")
            result = self.chain.invoke({"input": text})
            
            if not result or not isinstance(result, dict) or "intent" not in result:
                 raise ValueError(f"Invalid intent result (missing 'intent' key): {result}")
                 
            # Validate with Pydantic
            intent = IntentClassification(**result)
            print(f"Intent detected: {intent.intent} ({intent.confidence})")
            return intent
        except Exception as e:
            print(f"Intent classification failed: {e}")
            # Fallback to chat if uncertain
            return IntentClassification(intent="chat", confidence=0.0)

if __name__ == "__main__":
    agent = IntentAgent()
    print(agent.classify("Hello there"))
    print(agent.classify("Create a file called test.txt with hello world"))
