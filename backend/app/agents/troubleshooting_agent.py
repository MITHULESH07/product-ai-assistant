"""
Troubleshooting agent — analyzes faults and suggests fixes.
"""


def run(product: str, question: str) -> dict:
    return {
        "intent": "troubleshooting",
        "summary": f"Troubleshooting analysis for {product}.",
        "possible_causes": [],
        "steps": [],
        "warning": "",
        "escalation_required": False,
    }
