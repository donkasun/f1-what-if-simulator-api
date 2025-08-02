"""
Main FastAPI application entry point.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.v1.endpoints import router as api_v1_router
from app.core.config import settings
from app.core.exceptions import (
    DriverNotFoundError,
    InvalidSimulationParametersError,
    OpenF1APIError,
)
from app.core.logging_config import setup_logging

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager for startup and shutdown events."""
    # Startup
    logger.info("Starting F1 What-If Simulator API", version=app.version)
    yield
    # Shutdown
    logger.info("Shutting down F1 What-If Simulator API")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="F1 What-If Simulator API",
        description="A comprehensive API for Formula 1 simulation and analysis",
        version="0.1.0",
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
        lifespan=lifespan,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Request logging middleware
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", "unknown")
        logger.info(
            "Incoming request",
            request_id=request_id,
            method=request.method,
            url=str(request.url),
            client_ip=request.client.host if request.client else None,
        )

        response = await call_next(request)

        logger.info(
            "Request completed",
            request_id=request_id,
            status_code=response.status_code,
        )

        return response

    # Global exception handlers
    @app.exception_handler(DriverNotFoundError)
    async def driver_not_found_handler(request: Request, exc: DriverNotFoundError):
        logger.error(
            "Driver not found",
            request_id=request.headers.get("X-Request-ID", "unknown"),
            driver_id=exc.driver_id,
            exc_info=True,
        )
        return JSONResponse(
            status_code=404,
            content={
                "detail": f"Driver with ID {exc.driver_id} not found",
                "code": "DRIVER_NOT_FOUND",
            },
        )

    @app.exception_handler(InvalidSimulationParametersError)
    async def invalid_simulation_parameters_handler(
        request: Request, exc: InvalidSimulationParametersError
    ):
        logger.error(
            "Invalid simulation parameters",
            request_id=request.headers.get("X-Request-ID", "unknown"),
            error_details=exc.details,
            exc_info=True,
        )
        return JSONResponse(
            status_code=400,
            content={
                "detail": "Invalid simulation parameters provided",
                "code": "INVALID_SIMULATION_PARAMETERS",
                "details": exc.details,
            },
        )

    @app.exception_handler(OpenF1APIError)
    async def openf1_api_error_handler(request: Request, exc: OpenF1APIError):
        logger.error(
            "OpenF1 API error",
            request_id=request.headers.get("X-Request-ID", "unknown"),
            error_message=exc.message,
            status_code=exc.status_code,
            exc_info=True,
        )
        return JSONResponse(
            status_code=503,
            content={
                "detail": "External F1 data service temporarily unavailable",
                "code": "EXTERNAL_SERVICE_ERROR",
            },
        )

    # Include API routers
    app.include_router(api_v1_router, prefix="/api/v1")

    return app


# Create the application instance
app = create_app()

if __name__ == "__main__":
    import uvicorn

    setup_logging()
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    )
