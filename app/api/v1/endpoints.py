"""
API v1 endpoints for F1 What-If Simulator.
"""

from typing import List, Any

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
    CategoricalEncodingRequest,
    CategoricalMappingResponse,
    EncodingValidationResponse,
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
) -> Any:
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
) -> Any:
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
    """Get feature importance scores for a specific session."""
    logger.info("Getting feature importance", session_key=session_key)
    try:
        importance_scores = await simulation_service.get_feature_importance(session_key)
        return {
            "session_key": session_key,
            "feature_importance": importance_scores,
            "total_features": len(importance_scores),
            "top_features": sorted(
                importance_scores.items(), key=lambda x: x[1], reverse=True
            )[:10],
        }
    except FeatureEngineeringError as e:
        logger.error(
            "Feature engineering error while getting feature importance",
            session_key=session_key,
            error=str(e),
        )
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(
            "Failed to get feature importance",
            session_key=session_key,
            error=str(e),
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail="Failed to get feature importance")


@router.get("/feature-engineering/encoding-info/{session_key}", response_model=dict)
async def get_encoding_info(
    session_key: int,
    simulation_service: SimulationService = Depends(get_simulation_service),
) -> dict:
    """Get information about categorical encoding for a specific session."""
    logger.info("Getting encoding information", session_key=session_key)
    try:
        encoding_info = await simulation_service.get_encoding_info(session_key)
        return {
            "session_key": session_key,
            "encoding_info": encoding_info,
            "onehot_columns": encoding_info.get("onehot_columns", []),
            "label_columns": encoding_info.get("label_columns", []),
            "total_encoded_features": encoding_info.get("total_encoded_features", 0),
        }
    except FeatureEngineeringError as e:
        logger.error(
            "Feature engineering error while getting encoding info",
            session_key=session_key,
            error=str(e),
        )
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(
            "Failed to get encoding information",
            session_key=session_key,
            error=str(e),
            exc_info=True,
        )
        raise HTTPException(
            status_code=500, detail="Failed to get encoding information"
        )


@router.post("/feature-engineering/one-hot-encode", response_model=dict)
async def apply_one_hot_encoding(
    request: DataProcessingRequest,
    simulation_service: SimulationService = Depends(get_simulation_service),
) -> dict:
    """Apply one-hot encoding to categorical features for a session."""
    logger.info(
        "Applying one-hot encoding",
        session_key=request.session_key,
        processing_options=request.processing_options,
    )
    try:
        encoding_result = await simulation_service.apply_one_hot_encoding(request)
        return {
            "session_key": request.session_key,
            "encoding_result": encoding_result,
            "onehot_features_created": encoding_result.get(
                "onehot_features_created", 0
            ),
            "original_categorical_features": encoding_result.get(
                "original_categorical_features", []
            ),
            "new_feature_names": encoding_result.get("new_feature_names", []),
            "processing_time_ms": encoding_result.get("processing_time_ms", 0),
        }
    except FeatureEngineeringError as e:
        logger.error(
            "Feature engineering error while applying one-hot encoding",
            session_key=request.session_key,
            error=str(e),
        )
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(
            "Failed to apply one-hot encoding",
            session_key=request.session_key,
            error=str(e),
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail="Failed to apply one-hot encoding")


@router.get("/feature-engineering/encoding-stats/{session_key}", response_model=dict)
async def get_encoding_statistics(
    session_key: int,
    simulation_service: SimulationService = Depends(get_simulation_service),
) -> dict:
    """Get encoding statistics for a specific session."""
    logger.info("Getting encoding statistics", session_key=session_key)
    try:
        encoding_stats = await simulation_service.get_encoding_statistics(session_key)
        return {
            "session_key": session_key,
            "encoding_statistics": encoding_stats,
            "total_categorical_features": encoding_stats.get(
                "total_categorical_features", 0
            ),
            "onehot_encoded_features": encoding_stats.get("onehot_encoded_features", 0),
            "label_encoded_features": encoding_stats.get("label_encoded_features", 0),
            "feature_cardinality": encoding_stats.get("feature_cardinality", {}),
        }
    except FeatureEngineeringError as e:
        logger.error(
            "Feature engineering error while getting encoding statistics",
            session_key=session_key,
            error=str(e),
        )
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(
            "Failed to get encoding statistics",
            session_key=session_key,
            error=str(e),
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail="Failed to get encoding statistics")


# FWI-BE-106: Enhanced Categorical Encoding Endpoints


@router.post(
    "/feature-engineering/categorical/encode", response_model=CategoricalMappingResponse
)
async def encode_categorical_feature(
    request: CategoricalEncodingRequest,
    simulation_service: SimulationService = Depends(get_simulation_service),
) -> CategoricalMappingResponse:
    """Encode a specific categorical feature with enhanced mapping and validation."""
    logger.info(
        "Encoding categorical feature",
        session_key=request.session_key,
        feature_name=request.feature_name,
        encoding_type=request.encoding_type,
    )
    try:
        mapping_result = await simulation_service.encode_categorical_feature(request)
        return mapping_result
    except FeatureEngineeringError as e:
        logger.error(
            "Feature engineering error while encoding categorical feature",
            session_key=request.session_key,
            feature_name=request.feature_name,
            error=str(e),
        )
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(
            "Failed to encode categorical feature",
            session_key=request.session_key,
            feature_name=request.feature_name,
            error=str(e),
            exc_info=True,
        )
        raise HTTPException(
            status_code=500, detail="Failed to encode categorical feature"
        )


@router.get(
    "/feature-engineering/categorical/weather-mappings/{session_key}",
    response_model=dict,
)
async def get_weather_condition_mappings(
    session_key: int,
    simulation_service: SimulationService = Depends(get_simulation_service),
) -> dict:
    """Get weather condition categorical mappings for a session."""
    logger.info("Getting weather condition mappings", session_key=session_key)
    try:
        request = CategoricalEncodingRequest(
            session_key=session_key,
            feature_name="weather_condition",
            encoding_type="onehot",
        )
        mapping_result = await simulation_service.encode_categorical_feature(request)
        return {
            "session_key": session_key,
            "feature_name": "weather_condition",
            "mappings": mapping_result.feature_mappings,
            "categories": mapping_result.categories,
            "encoded_features": mapping_result.encoded_feature_names,
        }
    except Exception as e:
        logger.error(
            "Failed to get weather condition mappings",
            session_key=session_key,
            error=str(e),
            exc_info=True,
        )
        raise HTTPException(
            status_code=500, detail="Failed to get weather condition mappings"
        )


@router.get(
    "/feature-engineering/categorical/tire-mappings/{session_key}", response_model=dict
)
async def get_tire_compound_mappings(
    session_key: int,
    simulation_service: SimulationService = Depends(get_simulation_service),
) -> dict:
    """Get tire compound categorical mappings for a session."""
    logger.info("Getting tire compound mappings", session_key=session_key)
    try:
        request = CategoricalEncodingRequest(
            session_key=session_key,
            feature_name="tire_compound",
            encoding_type="onehot",
        )
        mapping_result = await simulation_service.encode_categorical_feature(request)
        return {
            "session_key": session_key,
            "feature_name": "tire_compound",
            "mappings": mapping_result.feature_mappings,
            "categories": mapping_result.categories,
            "encoded_features": mapping_result.encoded_feature_names,
        }
    except Exception as e:
        logger.error(
            "Failed to get tire compound mappings",
            session_key=session_key,
            error=str(e),
            exc_info=True,
        )
        raise HTTPException(
            status_code=500, detail="Failed to get tire compound mappings"
        )


@router.get(
    "/feature-engineering/categorical/track-type-mappings/{session_key}",
    response_model=dict,
)
async def get_track_type_mappings(
    session_key: int,
    simulation_service: SimulationService = Depends(get_simulation_service),
) -> dict:
    """Get track type categorical mappings for a session."""
    logger.info("Getting track type mappings", session_key=session_key)
    try:
        request = CategoricalEncodingRequest(
            session_key=session_key,
            feature_name="track_type",
            encoding_type="onehot",
        )
        mapping_result = await simulation_service.encode_categorical_feature(request)
        return {
            "session_key": session_key,
            "feature_name": "track_type",
            "mappings": mapping_result.feature_mappings,
            "categories": mapping_result.categories,
            "encoded_features": mapping_result.encoded_feature_names,
        }
    except Exception as e:
        logger.error(
            "Failed to get track type mappings",
            session_key=session_key,
            error=str(e),
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail="Failed to get track type mappings")


@router.get(
    "/feature-engineering/categorical/driver-team-mappings/{session_key}",
    response_model=dict,
)
async def get_driver_team_mappings(
    session_key: int,
    simulation_service: SimulationService = Depends(get_simulation_service),
) -> dict:
    """Get driver team categorical mappings for a session."""
    logger.info("Getting driver team mappings", session_key=session_key)
    try:
        request = CategoricalEncodingRequest(
            session_key=session_key,
            feature_name="driver_team",
            encoding_type="onehot",
        )
        mapping_result = await simulation_service.encode_categorical_feature(request)
        return {
            "session_key": session_key,
            "feature_name": "driver_team",
            "mappings": mapping_result.feature_mappings,
            "categories": mapping_result.categories,
            "encoded_features": mapping_result.encoded_feature_names,
        }
    except Exception as e:
        logger.error(
            "Failed to get driver team mappings",
            session_key=session_key,
            error=str(e),
            exc_info=True,
        )
        raise HTTPException(
            status_code=500, detail="Failed to get driver team mappings"
        )


@router.post(
    "/feature-engineering/categorical/validate",
    response_model=EncodingValidationResponse,
)
async def validate_categorical_encodings(
    request: DataProcessingRequest,
    simulation_service: SimulationService = Depends(get_simulation_service),
) -> EncodingValidationResponse:
    """Validate consistency of all categorical encodings for a session."""
    logger.info("Validating categorical encodings", session_key=request.session_key)
    try:
        validation_result = await simulation_service.validate_categorical_encodings(
            request
        )
        return validation_result
    except FeatureEngineeringError as e:
        logger.error(
            "Feature engineering error while validating encodings",
            session_key=request.session_key,
            error=str(e),
        )
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(
            "Failed to validate categorical encodings",
            session_key=request.session_key,
            error=str(e),
            exc_info=True,
        )
        raise HTTPException(
            status_code=500, detail="Failed to validate categorical encodings"
        )
