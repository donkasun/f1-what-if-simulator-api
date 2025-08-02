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
    StartingGridResponse,
    GridSummaryResponse,
    LapTimesResponse,
    PitStopsResponse,
    DriverPerformanceSummaryResponse,
    DataProcessingRequest,
    DataProcessingResponse,
)
from app.core.exceptions import (
    InvalidSimulationParametersError,
    FeatureEngineeringError,
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
        summary = await simulation_service.get_session_weather_summary(session_key)
        return summary
    except Exception as e:
        logger.error(
            "Failed to fetch weather summary",
            session_key=session_key,
            error=str(e),
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail="Failed to fetch weather summary")


@router.get("/grid/{session_key}", response_model=StartingGridResponse)
async def get_starting_grid(
    session_key: int,
    simulation_service: SimulationService = Depends(get_simulation_service),
) -> StartingGridResponse:
    """Get starting grid for a specific session."""
    logger.info("Fetching starting grid", session_key=session_key)
    try:
        grid = await simulation_service.get_starting_grid(session_key)
        return grid
    except Exception as e:
        logger.error(
            "Failed to fetch starting grid",
            session_key=session_key,
            error=str(e),
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail="Failed to fetch starting grid")


@router.get("/grid/{session_key}/summary", response_model=GridSummaryResponse)
async def get_grid_summary(
    session_key: int,
    simulation_service: SimulationService = Depends(get_simulation_service),
) -> GridSummaryResponse:
    """Get grid summary statistics for a specific session."""
    logger.info("Fetching grid summary", session_key=session_key)
    try:
        summary = await simulation_service.get_grid_summary(session_key)
        return summary
    except Exception as e:
        logger.error(
            "Failed to fetch grid summary",
            session_key=session_key,
            error=str(e),
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail="Failed to fetch grid summary")


@router.get("/lap-times/{session_key}", response_model=LapTimesResponse)
async def get_lap_times(
    session_key: int,
    simulation_service: SimulationService = Depends(get_simulation_service),
) -> LapTimesResponse:
    """Get lap times data for a specific session."""
    logger.info("Fetching lap times", session_key=session_key)
    try:
        lap_times_data = await simulation_service.get_lap_times(session_key)
        return lap_times_data
    except Exception as e:
        logger.error(
            "Failed to fetch lap times",
            session_key=session_key,
            error=str(e),
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail="Failed to fetch lap times")


@router.get("/pit-stops/{session_key}", response_model=PitStopsResponse)
async def get_pit_stops(
    session_key: int,
    simulation_service: SimulationService = Depends(get_simulation_service),
) -> PitStopsResponse:
    """Get pit stop data for a specific session."""
    logger.info("Fetching pit stops", session_key=session_key)
    try:
        pit_stops_data = await simulation_service.get_pit_stops(session_key)
        return pit_stops_data
    except Exception as e:
        logger.error(
            "Failed to fetch pit stops",
            session_key=session_key,
            error=str(e),
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail="Failed to fetch pit stops")


@router.get(
    "/driver-performance/{session_key}", response_model=DriverPerformanceSummaryResponse
)
async def get_driver_performance(
    session_key: int,
    simulation_service: SimulationService = Depends(get_simulation_service),
) -> DriverPerformanceSummaryResponse:
    """Get driver performance summary for a specific session."""
    logger.info("Fetching driver performance", session_key=session_key)
    try:
        performance_data = await simulation_service.get_driver_performance(session_key)
        return performance_data
    except Exception as e:
        logger.error(
            "Failed to fetch driver performance",
            session_key=session_key,
            error=str(e),
            exc_info=True,
        )
        raise HTTPException(
            status_code=500, detail="Failed to fetch driver performance"
        )


@router.post("/data-processing", response_model=DataProcessingResponse)
async def process_session_data(
    request: DataProcessingRequest,
    simulation_service: SimulationService = Depends(get_simulation_service),
) -> DataProcessingResponse:
    """Process and merge all session data into training-ready format."""
    logger.info(
        "Processing session data",
        session_key=request.session_key,
        include_weather=request.include_weather,
        include_grid=request.include_grid,
        include_lap_times=request.include_lap_times,
        include_pit_stops=request.include_pit_stops,
    )
    try:
        processed_data = await simulation_service.process_session_data(request)
        return processed_data
    except Exception as e:
        logger.error(
            "Failed to process session data",
            session_key=request.session_key,
            error=str(e),
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail="Failed to process session data")


@router.post("/feature-engineering/process", response_model=dict)
async def process_features(
    request: DataProcessingRequest,
    simulation_service: SimulationService = Depends(get_simulation_service),
) -> dict:
    """Process session data through the feature engineering pipeline."""
    logger.info("Processing features", session_key=request.session_key)
    try:
        # First get the processed data
        processed_response = await simulation_service.process_session_data(request)

        # Then apply feature engineering
        features, targets, metadata = (
            simulation_service.feature_engineering_service.fit_transform_features(
                processed_response.processed_data, target_column="lap_time"
            )
        )

        # Get feature importance scores
        feature_importance = (
            simulation_service.feature_engineering_service.get_feature_importance(
                "lap_time"
            )
        )

        # Get data quality report
        data_quality_report = (
            simulation_service.feature_engineering_service.get_data_quality_report(
                processed_response.processed_data
            )
        )

        return {
            "session_key": request.session_key,
            "features_shape": features.shape,
            "targets_shape": targets.shape,
            "feature_metadata": metadata,
            "feature_importance": feature_importance,
            "data_quality_report": data_quality_report,
            "processing_summary": processed_response.processing_summary.dict(),
        }

    except FeatureEngineeringError as e:
        logger.error("Feature engineering error", error=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(
            "Failed to process features",
            session_key=request.session_key,
            error=str(e),
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail="Failed to process features")


@router.get("/feature-engineering/quality-report/{session_key}", response_model=dict)
async def get_data_quality_report(
    session_key: int,
    simulation_service: SimulationService = Depends(get_simulation_service),
) -> dict:
    """Get a comprehensive data quality report for a session."""
    logger.info("Getting data quality report", session_key=session_key)
    try:
        # Get session data
        request = DataProcessingRequest(
            session_key=session_key,
            include_weather=True,
            include_grid=True,
            include_lap_times=True,
            include_pit_stops=True,
        )

        processed_response = await simulation_service.process_session_data(request)

        # Generate quality report
        quality_report = (
            simulation_service.feature_engineering_service.get_data_quality_report(
                processed_response.processed_data
            )
        )

        return {
            "session_key": session_key,
            "quality_report": quality_report,
            "processing_summary": processed_response.processing_summary.dict(),
        }

    except Exception as e:
        logger.error(
            "Failed to get data quality report",
            session_key=session_key,
            error=str(e),
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail="Failed to get data quality report")


@router.get(
    "/feature-engineering/feature-importance/{session_key}", response_model=dict
)
async def get_feature_importance(
    session_key: int,
    simulation_service: SimulationService = Depends(get_simulation_service),
) -> dict:
    """Get feature importance scores for a session."""
    logger.info("Getting feature importance", session_key=session_key)
    try:
        # Get session data and fit feature engineering pipeline
        request = DataProcessingRequest(
            session_key=session_key,
            include_weather=True,
            include_grid=True,
            include_lap_times=True,
            include_pit_stops=True,
        )

        processed_response = await simulation_service.process_session_data(request)

        # Fit the feature engineering pipeline
        features, targets, metadata = (
            simulation_service.feature_engineering_service.fit_transform_features(
                processed_response.processed_data, target_column="lap_time"
            )
        )

        # Get feature importance
        feature_importance = (
            simulation_service.feature_engineering_service.get_feature_importance(
                "lap_time"
            )
        )

        return {
            "session_key": session_key,
            "feature_importance": feature_importance,
            "feature_metadata": metadata,
            "total_features": len(feature_importance),
        }

    except FeatureEngineeringError as e:
        logger.error("Feature engineering error", error=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(
            "Failed to get feature importance",
            session_key=session_key,
            error=str(e),
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail="Failed to get feature importance")
