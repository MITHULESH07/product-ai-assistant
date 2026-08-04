import logging

from fastapi import APIRouter, HTTPException, Request, UploadFile

from app.schemas.analysis import AnalysisRequest, AnalysisResponse
from app.services.analysis_service import analyse_problem
from app.services.image_service import ImageServiceError, ProcessedImage, process_image

logger = logging.getLogger(__name__)

_VALID_ASSISTANCE_TYPES = frozenset({
    "troubleshooting",
    "operation",
    "maintenance",
})

router = APIRouter(tags=["AI Assistant"])


@router.post("/api/analyse", response_model=AnalysisResponse)
async def analyse(request: Request) -> AnalysisResponse:
    content_type = (request.headers.get("content-type") or "").lower()
    is_json = "application/json" in content_type

    if is_json:
        body = await request.json()
        question = _extract_json_field(body, "question", min_length=3, max_length=2000)
        product = _extract_json_field(body, "product", min_length=1, max_length=100)
        assistance_type = _extract_json_field(body, "assistance_type")
        _validate_assistance_type(assistance_type)
        processed_image = None
    else:
        form = await request.form()
        question = _extract_field(form, "question", min_length=3, max_length=2000)
        product = _extract_field(form, "product", min_length=1, max_length=100)
        assistance_type = _extract_field(form, "assistance_type")
        _validate_assistance_type(assistance_type)

        raw_file: UploadFile | None = form.get("file")
        processed_image = None
        if raw_file and raw_file.filename:
            _validate_image_content_type(raw_file)
            file_bytes = await raw_file.read()
            processed_image = process_image(
                file_bytes,
                raw_file.filename,
                raw_file.content_type,
            )
            logger.info(
                "Image attached — %s (%s) %dx%d",
                raw_file.filename,
                raw_file.content_type,
                processed_image.width,
                processed_image.height,
            )

    result = await analyse_problem(
        question=question,
        product=product,
        assistance_type=assistance_type,
        image=processed_image,
    )

    raw_sources = result.get("sources", [])
    sources: list[str] = []
    for s in raw_sources:
        if isinstance(s, str):
            sources.append(s)
        elif isinstance(s, dict):
            doc = s.get("document", s.get("source", ""))
            page = s.get("page", "")
            if page:
                sources.append(f"{doc} — page {page}")
            else:
                sources.append(doc)

    return AnalysisResponse(
        intent=str(result.get("intent", assistance_type)).strip(),
        product=str(result.get("product", product)).strip(),
        summary=str(result.get("summary", "")).strip(),
        possible_causes=[str(c).strip() for c in result.get("possible_causes", []) if c],
        steps=[str(s).strip() for s in result.get("steps", []) if s],
        warning=str(result.get("warning", "")).strip() or None,
        escalation_required=bool(result.get("escalation_required", False)),
        sources=sources,
    )


def _validate_assistance_type(value: str) -> None:
    if value not in _VALID_ASSISTANCE_TYPES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid assistance_type '{value}'. Must be one of: "
            f"{', '.join(sorted(_VALID_ASSISTANCE_TYPES))}.",
        )


def _validate_image_content_type(file: UploadFile) -> None:
    ct = (file.content_type or "").lower()
    if not ct.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file.content_type}'. Only images are accepted.",
        )


def _extract_field(
    form,
    name: str,
    min_length: int | None = None,
    max_length: int | None = None,
) -> str:
    value = form.get(name)
    if value is None:
        raise HTTPException(status_code=422, detail=f"Field '{name}' is required.")
    val = str(value).strip()
    if not val:
        raise HTTPException(
            status_code=422,
            detail=f"Field '{name}' must not be empty.",
        )
    if min_length is not None and len(val) < min_length:
        raise HTTPException(
            status_code=422,
            detail=f"Field '{name}' must be at least {min_length} characters.",
        )
    if max_length is not None and len(val) > max_length:
        raise HTTPException(
            status_code=422,
            detail=f"Field '{name}' must not exceed {max_length} characters.",
        )
    return val


def _extract_json_field(
    body: dict,
    name: str,
    min_length: int | None = None,
    max_length: int | None = None,
) -> str:
    value = body.get(name)
    if value is None or not isinstance(value, str) or not value.strip():
        raise HTTPException(
            status_code=422,
            detail=f"Field '{name}' is required.",
        )
    val = value.strip()
    if min_length is not None and len(val) < min_length:
        raise HTTPException(
            status_code=422,
            detail=f"Field '{name}' must be at least {min_length} characters.",
        )
    if max_length is not None and len(val) > max_length:
        raise HTTPException(
            status_code=422,
            detail=f"Field '{name}' must not exceed {max_length} characters.",
        )
    return val
