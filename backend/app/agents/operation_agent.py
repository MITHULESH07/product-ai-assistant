from app.clients import llm_factory
from app.services.image_service import ProcessedImage


class OperationAgent:
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
            assistance_type="operation",
            context=context,
            image=image,
        )