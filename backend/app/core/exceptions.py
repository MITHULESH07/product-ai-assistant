from fastapi import Request
from fastapi.responses import JSONResponse


class AnalysisError(Exception):
    def __init__(self, detail: str, status_code: int = 422):
        self.detail = detail
        self.status_code = status_code


class NotFoundError(Exception):
    def __init__(self, detail: str = "Resource not found"):
        self.detail = detail


def register_exception_handlers(app):
    @app.exception_handler(AnalysisError)
    async def analysis_error_handler(request: Request, exc: AnalysisError):
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail},
        )

    @app.exception_handler(NotFoundError)
    async def not_found_handler(request: Request, exc: NotFoundError):
        return JSONResponse(
            status_code=404,
            content={"detail": exc.detail},
        )

    @app.exception_handler(Exception)
    async def unexpected_error_handler(request: Request, exc: Exception):
        return JSONResponse(
            status_code=500,
            content={"detail": "An unexpected error occurred."},
        )
