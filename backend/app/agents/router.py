from app.llm.ollama_client import OllamaClient
from app.llm.prompts import build_prompt


class AgentRouter:

    def __init__(self):

        self.llm = OllamaClient()

    def route(self, question, product):

        prompt = build_prompt(
            question=question,
            product=product
        )

        answer = self.llm.generate(prompt)

        return {
            "intent": "general",
            "summary": answer
        }