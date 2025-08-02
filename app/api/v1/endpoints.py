"""
API v1 endpoints for F1 What-If Simulator.
"""

from typing import List

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.v1.schemas import (
    DriverResponse,
    SimulationRequest,
    SimulationResponse,
    TrackResponse,
    SessionResponse,
    WeatherDataResponse,
    WeatherSummaryResponse,
)
from app.core.exceptions import InvalidSimulationParametersError
from app.services.simulation_service import SimulationService

logger = structlog.get_logger()

router = APIRouter()


def get_simulation_service() -> SimulationService:
    """Dependency to get the simulation service instance."""
    return SimulationService()


@router.get("/health", response_model=dict)
async def health_check() -> dict:
    """Health check endpoint."""
    return {"status": "healthy", "service": "f1-what-if-simulator"}


@router.get("/drivers", response_model=List[DriverResponse])
async def get_drivers(
    season: int = Query(..., description="F1 season year"),
    simulation_service: SimulationService = Depends(get_simulation_service),
) -> List[DriverResponse]:
    """Get all drivers for a specific season."""
    logger.info("Fetching drivers", season=season)
    try:
        drivers = await simulation_service.get_drivers(season)
        return drivers
    except Exception as e:
        logger.error(
            "Failed to fetch drivers", season=season, error=str(e), exc_info=True
        )
        raise HTTPException(status_code=500, detail="Failed to fetch drivers")


@router.get("/tracks", response_model=List[TrackResponse])
async def get_tracks(
    season: int = Query(..., description="F1 season year"),
    simulation_service: SimulationService = Depends(get_simulation_service),
) -> List[TrackResponse]:
    """Get all tracks for a specific season."""
    logger.info("Fetching tracks", season=season)
    try:
        tracks = await simulation_service.get_tracks(season)
        return tracks
    except Exception as e:
        logger.error(
            "Failed to fetch tracks", season=season, error=str(e), exc_info=True
        )
        raise HTTPException(status_code=500, detail="Failed to fetch tracks")


@router.post("/simulate", response_model=SimulationResponse)
async def run_simulation(
    request: SimulationRequest,
    simulation_service: SimulationService = Depends(get_simulation_service),
) -> SimulationResponse:
    """Run a what-if simulation with the provided parameters."""
    logger.info(
        "Starting simulation",
        driver_id=request.driver_id,
        track_id=request.track_id,
        season=request.season,
    )
    try:
        result = await simulation_service.run_simulation(request)
        logger.info(
            "Simulation completed successfully",
            driver_id=request.driver_id,
            track_id=request.track_id,
            predicted_lap_time=result.predicted_lap_time,
        )
        return result
    except Exception as e:
        logger.error(
            "Simulation failed",
            driver_id=request.driver_id,
            track_id=request.track_id,
            error=str(e),
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail="Simulation failed")


@router.get("/simulation/{simulation_id}", response_model=SimulationResponse)
async def get_simulation_result(
    simulation_id: str,
    simulation_service: SimulationService = Depends(get_simulation_service),
) -> SimulationResponse:
    """Get the result of a previously run simulation."""
    logger.info("Fetching simulation result", simulation_id=simulation_id)
    try:
        result = await simulation_service.get_simulation_result(simulation_id)
        logger.info(
            "Successfully fetched simulation result", simulation_id=simulation_id
        )
        return result
    except InvalidSimulationParametersError as e:
        logger.error(
            "Failed to fetch simulation result",
            simulation_id=simulation_id,
            error=str(e),
        )
        raise HTTPException(status_code=404, detail="Simulation not found")
    except Exception as e:
        logger.error(
            "Failed to fetch simulation result",
            simulation_id=simulation_id,
            error=str(e),
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail="Failed to fetch simulation result")


@router.get("/cache/stats", response_model=dict)
async def get_cache_stats(
    simulation_service: SimulationService = Depends(get_simulation_service),
) -> dict:
    """Get cache statistics for debugging and monitoring."""
    return simulation_service.get_cache_stats()


@router.delete("/cache/clear")
async def clear_cache(
    simulation_service: SimulationService = Depends(get_simulation_service),
) -> dict:
    """Clear all cached simulation results."""
    simulation_service.clear_cache()
    return {"message": "Cache cleared successfully"}


@router.delete("/cache/{simulation_id}")
async def remove_from_cache(
    simulation_id: str,
    simulation_service: SimulationService = Depends(get_simulation_service),
) -> dict:
    """Remove a specific simulation from cache."""
    removed = simulation_service.remove_from_cache(simulation_id)
    if removed:
        return {"message": f"Simulation {simulation_id} removed from cache"}
    else:
        raise HTTPException(status_code=404, detail="Simulation not found in cache")


@router.get("/sessions", response_model=List[SessionResponse])
async def get_sessions(
    season: int = Query(..., description="F1 season year"),
    simulation_service: SimulationService = Depends(get_simulation_service),
) -> List[SessionResponse]:
    """Get all sessions for a specific season."""
    logger.info("Fetching sessions", season=season)
    try:
        sessions = await simulation_service.get_sessions(season)
        return sessions
    except Exception as e:
        logger.error(
            "Failed to fetch sessions", season=season, error=str(e), exc_info=True
        )
        raise HTTPException(status_code=500, detail="Failed to fetch sessions")


@router.get("/weather/{session_key}", response_model=List[WeatherDataResponse])
async def get_weather_data(
    session_key: int,
    simulation_service: SimulationService = Depends(get_simulation_service),
) -> List[WeatherDataResponse]:
    """Get weather data for a specific session."""
    logger.info("Fetching weather data", session_key=session_key)
    try:
        weather_data = await simulation_service.get_weather_data(session_key)
        return weather_data
    except Exception as e:
        logger.error(
            "Failed to fetch weather data",
            session_key=session_key,
            error=str(e),
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail="Failed to fetch weather data")


@router.get("/weather/{session_key}/summary", response_model=WeatherSummaryResponse)
async def get_weather_summary(
    session_key: int,
    simulation_service: SimulationService = Depends(get_simulation_service),
) -> WeatherSummaryResponse:
    """Get weather summary for a specific session."""
    logger.info("Fetching weather summary", session_key=session_key)
    try:
        weather_summary = await simulation_service.get_session_weather_summary(
            session_key
        )
        return weather_summary
    except Exception as e:
        logger.error(
            "Failed to fetch weather summary",
            session_key=session_key,
            error=str(e),
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail="Failed to fetch weather summary")
