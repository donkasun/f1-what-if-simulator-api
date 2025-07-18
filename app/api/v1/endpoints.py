"""
API v1 endpoints for F1 What-If Simulator.
"""

from typing import List

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse

from app.api.v1.schemas import (
    DriverResponse,
    SimulationRequest,
    SimulationResponse,
    TrackResponse,
)
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
        logger.error("Failed to fetch drivers", season=season, error=str(e), exc_info=True)
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
        logger.error("Failed to fetch tracks", season=season, error=str(e), exc_info=True)
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
        return result
    except Exception as e:
        logger.error(
            "Failed to fetch simulation result",
            simulation_id=simulation_id,
            error=str(e),
            exc_info=True,
        )
        raise HTTPException(status_code=404, detail="Simulation not found") 