from app.clients import llm_factory
from app.services.image_service import ProcessedImage


class TroubleshootingAgent:
    async def run(
        self,
        product: str,
        question: str,
        context: str | None = None,
        image: ProcessedImage | None = None,
    ) -> dict:
        return await llm_factory.generate_assistance(
            question=question,
            product=product,
            assistance_type="troubleshooting",
            context=context,
            image=image,
        )