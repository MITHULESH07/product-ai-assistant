"""
Maintenance agent — recommends upkeep and preventive care.
"""


def run(product: str, question: str) -> dict:
    return {
        "intent": "maintenance",
        "summary": f"Maintenance recommendations for {product}.",
        "possible_causes": [],
        "steps": [],
        "warning": "",
        "escalation_required": False,
    }
