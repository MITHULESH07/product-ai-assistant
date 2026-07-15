"""
Operation agent — guides users on correct product usage.
"""


def run(product: str, question: str) -> dict:
    return {
        "intent": "operation",
        "summary": f"Operation guidance for {product}.",
        "possible_causes": [],
        "steps": [],
        "warning": "",
        "escalation_required": False,
    }
