"""
Structured logging configuration for F1 What-If Simulator API.
"""

import logging
import sys
from typing import Any, Dict

import structlog
from structlog.stdlib import LoggerFactory

from app.core.config import settings


def setup_logging() -> None:
    """Configure structured logging for the application."""

    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            (
                structlog.processors.JSONRenderer()
                if settings.log_format == "json"
                else structlog.dev.ConsoleRenderer()
            ),
        ],
        context_class=dict,
        logger_factory=LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.log_level.upper()),
    )

    # Set specific log levels for noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)


def get_logger(name: str = "") -> structlog.BoundLogger:
    """Get a structured logger instance.

    Args:
        name: Logger name (optional)

    Returns:
        Configured structured logger
    """
    logger: structlog.BoundLogger = structlog.get_logger(name)
    return logger


def log_request_info(
    logger: structlog.BoundLogger, request_info: Dict[str, Any]
) -> None:
    """Log standardized request information.

    Args:
        logger: Structured logger instance
        request_info: Dictionary containing request details
    """
    logger.info(
        "HTTP request",
        method=request_info.get("method"),
        url=request_info.get("url"),
        client_ip=request_info.get("client_ip"),
        user_agent=request_info.get("user_agent"),
        request_id=request_info.get("request_id"),
    )


def log_response_info(
    logger: structlog.BoundLogger, response_info: Dict[str, Any]
) -> None:
    """Log standardized response information.

    Args:
        logger: Structured logger instance
        response_info: Dictionary containing response details
    """
    logger.info(
        "HTTP response",
        status_code=response_info.get("status_code"),
        response_time_ms=response_info.get("response_time_ms"),
        request_id=response_info.get("request_id"),
    )


def log_external_api_call(
    logger: structlog.BoundLogger,
    api_name: str,
    endpoint: str,
    method: str,
    status_code: int,
    response_time_ms: int,
    error: str = "",
) -> None:
    """Log external API call information.

    Args:
        logger: Structured logger instance
        api_name: Name of the external API
        endpoint: API endpoint
        method: HTTP method
        status_code: Response status code
        response_time_ms: Response time in milliseconds
        error: Error message if the call failed
    """
    log_data = {
        "api_name": api_name,
        "endpoint": endpoint,
        "method": method,
        "status_code": status_code,
        "response_time_ms": response_time_ms,
    }

    if error:
        logger.error("External API call failed", **log_data, error=error)
    else:
        logger.info("External API call completed", **log_data)


def log_simulation_event(
    logger: structlog.BoundLogger,
    event_type: str,
    simulation_id: str,
    driver_id: int,
    track_id: int,
    season: int,
    **kwargs,
) -> None:
    """Log simulation-related events.

    Args:
        logger: Structured logger instance
        event_type: Type of simulation event
        simulation_id: Unique simulation identifier
        driver_id: Driver ID
        track_id: Track ID
        season: F1 season
        **kwargs: Additional event-specific data
    """
    logger.info(
        f"Simulation {event_type}",
        simulation_id=simulation_id,
        driver_id=driver_id,
        track_id=track_id,
        season=season,
        **kwargs,
    )
