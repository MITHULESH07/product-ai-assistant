import logging

logger = logging.getLogger(__name__)


def analyse_problem(
    question: str,
    product: str,
    assistance_type: str = "auto",
) -> dict:
    logger.info(
        "Analysis requested — product=%s type=%s",
        product,
        assistance_type,
    )

    return {
        "intent": "troubleshooting",
        "product": product,
        "summary": (
            f"Received your question about the {product}: "
            f"{question}"
        ),
        "possible_causes": [
            "Loose wiring or poor connection",
            "Incorrect configuration or firmware issue",
            "Component wear or environmental stress",
        ],
        "steps": [
            "Check all physical connections and power supply.",
            "Review the product manual for correct setup.",
            f"Test the {product} with a minimal configuration.",
        ],
        "warning": "",
        "escalation_required": False,
        "sources": [
            {
                "document": f"{product} User Manual",
                "page": 12,
            },
            {
                "document": f"{product} Troubleshooting Guide",
                "page": 5,
            },
        ],
        "assistance_type": assistance_type,
    }


def route_request(
    question: str,
    product: str,
    assistance_type: str,
) -> dict:
    """
    Future integration point for agent routing, RAG retrieval,
    OCR processing, and LLM-based response generation.

    Currently delegates to the dummy analysis service.
    """
    return analyse_problem(
        question=question,
        product=product,
        assistance_type=assistance_type,
    )
