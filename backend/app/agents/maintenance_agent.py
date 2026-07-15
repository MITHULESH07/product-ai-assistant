from app.clients.ollama_client import OllamaClient
from app.prompts.maintenance import build_maintenance_prompt


class MaintenanceAgent:
    def __init__(self) -> None:
        self.ollama_client = OllamaClient()

    async def run(
        self,
        product: str,
        question: str,
    ) -> str:
        prompt = build_maintenance_prompt(
            product=product,
            question=question,
        )

        return await self.ollama_client.generate(prompt)