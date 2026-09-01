from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.router import api_router
from app.core.config import get_settings
from app.core.exceptions import InsightPDFError
from app.core.logging import get_logger, setup_logging
from app.database import init_db
from app.tools.registry import ensure_loaded

setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info("Starting %s (%s)", settings.APP_NAME, settings.ENVIRONMENT)
    init_db()
    ensure_loaded()
    logger.info("Tool registry loaded, database ready.")
    yield
    logger.info("Shutting down.")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=f"{settings.APP_NAME} API",
        description="Agentic RAG Document Intelligence Platform",
        version="2.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(InsightPDFError)
    async def handle_app_error(request: Request, exc: InsightPDFError):
        logger.warning("%s: %s (%s)", type(exc).__name__, exc.user_message, request.url.path)
        return JSONResponse(status_code=exc.http_status, content={"error": type(exc).__name__, "message": exc.user_message})

    @app.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, exc: Exception):
        # Any exception that isn't one of our typed InsightPDFError
        # subclasses is a bug, not an anticipated failure mode -- log the
        # full traceback server-side, but never leak it (or a raw
        # "Internal Server Error") to the client. The frontend always gets
        # the same structured {error, message} shape either way.
        logger.exception("Unhandled exception on %s", request.url.path)
        return JSONResponse(
            status_code=500,
            content={
                "error": "InternalError",
                "message": "Something went wrong processing that request. Please try again.",
            },
        )

    @app.get("/health")
    def health():
        return {"status": "ok"}

    app.include_router(api_router, prefix=settings.API_PREFIX)
    return app


app = create_app()
