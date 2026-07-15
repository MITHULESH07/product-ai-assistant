"""
Routes analysis requests to the appropriate agent based on assistance type.
"""


def route_to_agent(question: str, product: str, assistance_type: str) -> dict:
    """
    Placeholder — routes to the correct agent implementation.
    Returns a dict matching AnalysisResponse fields.
    """
    return {
        "intent": assistance_type if assistance_type != "auto" else "general",
        "product": product,
        "summary": f"Routing to {assistance_type} agent.",
        "possible_causes": [],
        "steps": [],
        "warning": "",
        "escalation_required": False,
        "sources": [],
    }
