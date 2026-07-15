from app.clients.ollama_client import OllamaClient
from app.prompts.operation import build_operation_prompt


class OperationAgent:
    def __init__(self) -> None:
        self.ollama_client = OllamaClient()

    async def run(
        self,
        product: str,
        question: str,
    ) -> str:
        prompt = build_operation_prompt(
            product=product,
            question=question,
        )

        return await self.ollama_client.generate(prompt)