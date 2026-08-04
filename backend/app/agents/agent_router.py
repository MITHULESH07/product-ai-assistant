"""
Routes analysis requests to the appropriate agent based on assistance type.
"""

from app.agents.maintenance_agent import MaintenanceAgent
from app.agents.operation_agent import OperationAgent
from app.agents.troubleshooting_agent import TroubleshootingAgent
from app.services.image_service import ProcessedImage


async def route_to_agent(
    question: str,
    product: str,
    assistance_type: str,
    context: str | None = None,
    image: ProcessedImage | None = None,
) -> dict:
    if assistance_type == "troubleshooting":
        agent = TroubleshootingAgent()
    elif assistance_type == "operation":
        agent = OperationAgent()
    elif assistance_type == "maintenance":
        agent = MaintenanceAgent()
    else:
        agent = TroubleshootingAgent()

    return await agent.run(
        product=product,
        question=question,
        context=context,
        image=image,
    )
