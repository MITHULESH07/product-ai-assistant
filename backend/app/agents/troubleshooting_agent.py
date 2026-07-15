from app.clients.ollama_client import OllamaClient
from app.prompts.troubleshooting import (
    build_troubleshooting_prompt,
)


class TroubleshootingAgent:
    def __init__(self) -> None:
        self.ollama_client = OllamaClient()

    async def run(
        self,
        product: str,
        question: str,
    ) -> str:
        prompt = build_troubleshooting_prompt(
            product=product,
            question=question,
        )

        return await self.ollama_client.generate(prompt)