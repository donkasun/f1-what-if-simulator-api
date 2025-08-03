"""
Simulation service for F1 What-If Simulator.

This module provides business logic for running F1 simulations,
including ML model integration and result caching.
"""

import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import structlog
from async_lru import alru_cache

from app.core.exceptions import (
    DriverNotFoundError,
    InvalidSimulationParametersError,
)
from app.external.openf1_client import OpenF1Client
from app.models.model_loader import ModelLoader
from app.services.feature_engineering_service import FeatureEngineeringService
from app.api.v1.schemas import (
    DriverResponse,
    SimulationRequest,
    SimulationResponse,
    TrackResponse,
    SessionResponse,
    WeatherDataResponse,
    WeatherSummaryResponse,
    StartingGridResponse,
    GridPositionResponse,
    GridSummaryResponse,
    LapTimesResponse,
    LapTimeResponse,
    PitStopsResponse,
    PitStopResponse,
    DriverPerformanceSummaryResponse,
    DriverPerformanceResponse,
    DataProcessingRequest,
    DataProcessingResponse,
    DataProcessingSummary,
    ProcessedDataPoint,
    CategoricalEncodingRequest,
    CategoricalMappingResponse,
    EncodingValidationResponse,
)

logger = structlog.get_logger()

# Global cache for simulation results - persists across service instances
_simulation_cache: dict[str, Any] = {}


class SimulationService:
    """Service class for handling F1 simulation business logic."""

    def __init__(self):
        """Initialize the simulation service with dependencies."""
        self.openf1_client = OpenF1Client()
        self.model_loader = ModelLoader()
        self.feature_engineering_service = FeatureEngineeringService()
        # Use the global cache instead of instance-level cache
        self._simulation_cache = _simulation_cache

    @alru_cache(maxsize=100)
    async def get_drivers(self, season: int) -> List[DriverResponse]:
        """
        Get all drivers for a specific season.

        Args:
            season: F1 season year

        Returns:
            List of driver information
        """
        logger.info("Fetching drivers for season", season=season)
        async with self.openf1_client as client:
            drivers_data = await client.get_drivers(season=season)
            drivers = [
                DriverResponse(
                    driver_id=driver.get("driver_number"),
                    name=driver.get("full_name"),
                    code=driver.get("name_acronym"),
                    team=driver.get("team_name"),
                    nationality=driver.get("country_code"),
                )
                for driver in drivers_data
            ]
            logger.info(
                "Successfully fetched drivers", season=season, count=len(drivers)
            )
            return drivers

    @alru_cache(maxsize=50)
    async def get_tracks(self, season: int) -> List[TrackResponse]:
        """
        Get all tracks for a specific season.

        Args:
            season: F1 season year

        Returns:
            List of track information
        """
        logger.info("Fetching tracks for season", season=season)
        async with self.openf1_client as client:
            tracks_data = await client.get_tracks(season=season)
            tracks = [
                TrackResponse(
                    track_id=track.get("track_id"),
                    name=track.get("name"),
                    country=track.get("country"),
                    circuit_length=track.get("circuit_length"),
                    number_of_laps=track.get("number_of_laps"),
                )
                for track in tracks_data
            ]
            logger.info("Successfully fetched tracks", season=season, count=len(tracks))
            return tracks

    async def get_sessions(self, season: int) -> List[SessionResponse]:
        """Get all sessions for a specific season."""
        logger.info("Fetching sessions", season=season)

        async with self.openf1_client as client:
            sessions_data = await client.get_sessions(season)

            sessions = []
            for session in sessions_data:
                sessions.append(
                    SessionResponse(
                        session_key=session["session_key"],
                        meeting_key=session["meeting_key"],
                        location=session["location"],
                        session_type=session["session_type"],
                        session_name=session["session_name"],
                        date_start=session["date_start"],
                        date_end=session["date_end"],
                        country_name=session["country_name"],
                        circuit_short_name=session["circuit_short_name"],
                        year=session["year"],
                    )
                )

            return sessions

    async def get_weather_data(self, session_key: int) -> List[WeatherDataResponse]:
        """Get weather data for a specific session."""
        logger.info("Fetching weather data", session_key=session_key)

        async with self.openf1_client as client:
            weather_data = await client.get_weather_data(session_key)

            weather_responses = []
            for weather in weather_data:
                weather_responses.append(
                    WeatherDataResponse(
                        date=weather["date"],
                        session_key=weather["session_key"],
                        air_temperature=weather["air_temperature"],
                        track_temperature=weather["track_temperature"],
                        humidity=weather["humidity"],
                        pressure=weather["pressure"],
                        wind_speed=weather["wind_speed"],
                        wind_direction=weather["wind_direction"],
                        rainfall=weather["rainfall"],
                    )
                )

            return weather_responses

    async def get_session_weather_summary(
        self, session_key: int
    ) -> WeatherSummaryResponse:
        """Get weather summary for a specific session."""
        logger.info("Fetching weather summary", session_key=session_key)

        async with self.openf1_client as client:
            weather_summary = await client.get_session_weather_summary(session_key)

            return WeatherSummaryResponse(
                session_key=weather_summary["session_key"],
                weather_condition=weather_summary["weather_condition"],
                avg_air_temperature=weather_summary["avg_air_temperature"],
                avg_track_temperature=weather_summary["avg_track_temperature"],
                avg_humidity=weather_summary["avg_humidity"],
                avg_pressure=weather_summary["avg_pressure"],
                avg_wind_speed=weather_summary["avg_wind_speed"],
                total_rainfall=weather_summary["total_rainfall"],
                data_points=weather_summary["data_points"],
            )

    async def get_starting_grid(self, session_key: int) -> StartingGridResponse:
        """
        Get the starting grid for a specific session.

        Args:
            session_key: Session identifier

        Returns:
            Starting grid data with all positions

        Raises:
            OpenF1APIError: If API call fails
        """
        logger.info("Fetching starting grid", session_key=session_key)

        async with self.openf1_client as client:
            grid_data = await client.get_starting_grid(session_key)
            grid_summary = await client.get_session_grid_summary(session_key)

            # Get session info for additional context
            sessions = await client.get_sessions(grid_summary.get("year", 2024))
            session_info = None
            for session in sessions:
                if session.get("session_key") == session_key:
                    session_info = session
                    break

            grid_positions = []
            for position_data in grid_data:
                grid_positions.append(
                    GridPositionResponse(
                        position=position_data.get("position"),
                        driver_id=position_data.get("driver_id"),
                        driver_name=position_data.get("driver_name"),
                        driver_code=position_data.get("driver_code"),
                        team_name=position_data.get("team_name"),
                        qualifying_time=position_data.get("qualifying_time"),
                        qualifying_gap=position_data.get("qualifying_gap"),
                        qualifying_laps=position_data.get("qualifying_laps"),
                    )
                )

            return StartingGridResponse(
                session_key=session_key,
                session_name=(
                    session_info.get("session_name")
                    if session_info
                    else f"Session {session_key}"
                ),
                track_name=(
                    session_info.get("circuit_short_name")
                    if session_info
                    else "Unknown Track"
                ),
                country=session_info.get("country_name") if session_info else "Unknown",
                year=session_info.get("year") if session_info else 2024,
                total_drivers=len(grid_positions),
                grid_positions=grid_positions,
            )

    async def get_grid_summary(self, session_key: int) -> GridSummaryResponse:
        """
        Get grid summary statistics for a specific session.

        Args:
            session_key: Session identifier

        Returns:
            Grid summary with statistics

        Raises:
            OpenF1APIError: If API call fails
        """
        logger.info("Fetching grid summary", session_key=session_key)

        async with self.openf1_client as client:
            grid_summary = await client.get_session_grid_summary(session_key)

            # Convert pole position data to GridPositionResponse if available
            pole_position = None
            if grid_summary.get("pole_position"):
                pole_data = grid_summary["pole_position"]
                pole_position = GridPositionResponse(
                    position=pole_data.get("position"),
                    driver_id=pole_data.get("driver_id"),
                    driver_name=pole_data.get("driver_name"),
                    driver_code=pole_data.get("driver_code"),
                    team_name=pole_data.get("team_name"),
                    qualifying_time=pole_data.get("qualifying_time"),
                    qualifying_gap=pole_data.get("qualifying_gap"),
                    qualifying_laps=pole_data.get("qualifying_laps"),
                )

            return GridSummaryResponse(
                session_key=session_key,
                pole_position=pole_position,
                fastest_qualifying_time=grid_summary.get("fastest_qualifying_time"),
                slowest_qualifying_time=grid_summary.get("slowest_qualifying_time"),
                average_qualifying_time=grid_summary.get("average_qualifying_time"),
                time_gap_pole_to_last=grid_summary.get("time_gap_pole_to_last"),
                teams_represented=grid_summary.get("teams_represented", []),
            )

    async def get_lap_times(self, session_key: int) -> LapTimesResponse:
        """
        Get lap times data for a specific session.

        Args:
            session_key: Session identifier

        Returns:
            Lap times data with session information
        """
        logger.info("Fetching lap times", session_key=session_key)

        async with self.openf1_client as client:
            lap_times_summary = await client.get_session_lap_times_summary(session_key)

            # Convert to response schema
            lap_times = []
            for lap_data in lap_times_summary.get("lap_times", []):
                lap_time = LapTimeResponse(
                    lap_number=lap_data.get("lap_number"),
                    driver_id=lap_data.get("driver_id"),
                    driver_name=lap_data.get("driver_name"),
                    driver_code=lap_data.get("driver_code"),
                    team_name=lap_data.get("team_name"),
                    lap_time=lap_data.get("lap_time"),
                    sector_1_time=lap_data.get("sector_1_time"),
                    sector_2_time=lap_data.get("sector_2_time"),
                    sector_3_time=lap_data.get("sector_3_time"),
                    tire_compound=lap_data.get("tire_compound"),
                    fuel_load=lap_data.get("fuel_load"),
                    lap_status=lap_data.get("lap_status"),
                    timestamp=datetime.fromisoformat(
                        lap_data.get("timestamp").replace("Z", "+00:00")
                    ),
                )
                lap_times.append(lap_time)

            return LapTimesResponse(
                session_key=lap_times_summary.get("session_key"),
                session_name=lap_times_summary.get("session_name"),
                track_name=lap_times_summary.get("track_name"),
                country=lap_times_summary.get("country"),
                year=lap_times_summary.get("year"),
                total_laps=lap_times_summary.get("total_laps"),
                lap_times=lap_times,
            )

    async def get_pit_stops(self, session_key: int) -> PitStopsResponse:
        """
        Get pit stop data for a specific session.

        Args:
            session_key: Session identifier

        Returns:
            Pit stop data with session information
        """
        logger.info("Fetching pit stops", session_key=session_key)

        async with self.openf1_client as client:
            pit_stops_summary = await client.get_session_pit_stops_summary(session_key)

            # Convert to response schema
            pit_stops = []
            for pit_data in pit_stops_summary.get("pit_stops", []):
                pit_stop = PitStopResponse(
                    pit_stop_number=pit_data.get("pit_stop_number"),
                    driver_id=pit_data.get("driver_id"),
                    driver_name=pit_data.get("driver_name"),
                    driver_code=pit_data.get("driver_code"),
                    team_name=pit_data.get("team_name"),
                    lap_number=pit_data.get("lap_number"),
                    pit_duration=pit_data.get("pit_duration"),
                    tire_compound_in=pit_data.get("tire_compound_in"),
                    tire_compound_out=pit_data.get("tire_compound_out"),
                    fuel_added=pit_data.get("fuel_added"),
                    pit_reason=pit_data.get("pit_reason"),
                    timestamp=datetime.fromisoformat(
                        pit_data.get("timestamp").replace("Z", "+00:00")
                    ),
                )
                pit_stops.append(pit_stop)

            return PitStopsResponse(
                session_key=pit_stops_summary.get("session_key"),
                session_name=pit_stops_summary.get("session_name"),
                track_name=pit_stops_summary.get("track_name"),
                country=pit_stops_summary.get("country"),
                year=pit_stops_summary.get("year"),
                total_pit_stops=pit_stops_summary.get("total_pit_stops"),
                pit_stops=pit_stops,
            )

    async def get_driver_performance(
        self, session_key: int
    ) -> DriverPerformanceSummaryResponse:
        """
        Get driver performance summary for a specific session.

        Args:
            session_key: Session identifier

        Returns:
            Driver performance data with session information
        """
        logger.info("Fetching driver performance", session_key=session_key)

        async with self.openf1_client as client:
            performance_summary = await client.get_session_driver_performance_summary(
                session_key
            )

            # Convert to response schema
            driver_performances = []
            for perf_data in performance_summary.get("driver_performances", []):
                performance = DriverPerformanceResponse(
                    driver_id=perf_data.get("driver_id"),
                    driver_name=perf_data.get("driver_name"),
                    driver_code=perf_data.get("driver_code"),
                    team_name=perf_data.get("team_name"),
                    total_laps=perf_data.get("total_laps"),
                    best_lap_time=perf_data.get("best_lap_time"),
                    avg_lap_time=perf_data.get("avg_lap_time"),
                    consistency_score=perf_data.get("consistency_score"),
                    total_pit_stops=perf_data.get("total_pit_stops"),
                    total_pit_time=perf_data.get("total_pit_time"),
                    avg_pit_time=perf_data.get("avg_pit_time"),
                    tire_compounds_used=perf_data.get("tire_compounds_used", []),
                    final_position=perf_data.get("final_position"),
                    race_status=perf_data.get("race_status"),
                )
                driver_performances.append(performance)

            return DriverPerformanceSummaryResponse(
                session_key=performance_summary.get("session_key"),
                session_name=performance_summary.get("session_name"),
                track_name=performance_summary.get("track_name"),
                country=performance_summary.get("country"),
                year=performance_summary.get("year"),
                total_drivers=performance_summary.get("total_drivers"),
                driver_performances=driver_performances,
            )

    async def run_simulation(self, request: SimulationRequest) -> SimulationResponse:
        """
        Run a what-if simulation with the provided parameters.

        Args:
            request: Simulation parameters

        Returns:
            Simulation results

        Raises:
            DriverNotFoundError: If driver is not found
            InvalidSimulationParametersError: If parameters are invalid
        """
        start_time = time.time()
        simulation_id = f"sim_{uuid.uuid4().hex[:12]}"

        logger.info(
            "Starting simulation",
            simulation_id=simulation_id,
            driver_id=request.driver_id,
            track_id=request.track_id,
            season=request.season,
        )

        try:
            # Validate that driver exists
            drivers = await self.get_drivers(request.season)
            driver_exists = any(d.driver_id == request.driver_id for d in drivers)
            if not driver_exists:
                raise DriverNotFoundError(request.driver_id)

            # Validate that track exists
            tracks = await self.get_tracks(request.season)
            track_exists = any(t.track_id == request.track_id for t in tracks)
            if not track_exists:
                raise InvalidSimulationParametersError(
                    f"Track with ID {request.track_id} not found"
                )

            # Get historical data for the driver and track
            historical_data = await self._get_historical_data(
                driver_id=request.driver_id,
                track_id=request.track_id,
                season=request.season,
            )

            # Load and run the ML model
            model = await self.model_loader.get_model()
            prediction_features = self._prepare_features(request, historical_data)
            predicted_lap_time = model.predict([prediction_features])[0]

            # Calculate confidence score based on data quality
            confidence_score = self._calculate_confidence_score(historical_data)

            processing_time_ms = int((time.time() - start_time) * 1000)

            # Create response
            result = SimulationResponse(
                simulation_id=simulation_id,
                driver_id=request.driver_id,
                track_id=request.track_id,
                season=request.season,
                predicted_lap_time=float(predicted_lap_time),
                confidence_score=confidence_score,
                weather_conditions=request.weather_conditions or "dry",
                car_setup=request.car_setup or {},
                created_at=datetime.now(timezone.utc),
                processing_time_ms=processing_time_ms,
            )

            # Cache the result
            self._simulation_cache[simulation_id] = result

            logger.info(
                "Simulation completed successfully",
                simulation_id=simulation_id,
                predicted_lap_time=result.predicted_lap_time,
                confidence_score=result.confidence_score,
                processing_time_ms=processing_time_ms,
            )

            response: SimulationResponse = result
            return response

        except (DriverNotFoundError, InvalidSimulationParametersError):
            # Re-raise business exceptions
            raise
        except Exception as e:
            logger.error(
                "Simulation failed unexpectedly",
                simulation_id=simulation_id,
                error=str(e),
                exc_info=True,
            )
            raise InvalidSimulationParametersError(f"Simulation failed: {str(e)}")

    async def get_simulation_result(self, simulation_id: str) -> SimulationResponse:
        """
        Get the result of a previously run simulation.

        Args:
            simulation_id: Unique simulation identifier

        Returns:
            Cached simulation result

        Raises:
            InvalidSimulationParametersError: If simulation not found
        """
        if simulation_id not in self._simulation_cache:
            raise InvalidSimulationParametersError(
                f"Simulation {simulation_id} not found"
            )

        result: SimulationResponse = self._simulation_cache[simulation_id]
        return result

    def get_cache_stats(self) -> dict:
        """
        Get cache statistics for debugging and monitoring.

        Returns:
            Dictionary with cache statistics
        """
        return {
            "cache_size": len(self._simulation_cache),
            "cached_simulations": list(self._simulation_cache.keys()),
            "cache_hits": getattr(self, "_cache_hits", 0),
            "cache_misses": getattr(self, "_cache_misses", 0),
        }

    def clear_cache(self) -> None:
        """Clear all cached simulation results."""
        self._simulation_cache.clear()
        logger.info("Simulation cache cleared")

    def remove_from_cache(self, simulation_id: str) -> bool:
        """
        Remove a specific simulation result from cache.

        Args:
            simulation_id: Unique simulation identifier

        Returns:
            True if removed, False if not found
        """
        if simulation_id in self._simulation_cache:
            del self._simulation_cache[simulation_id]
            logger.info("Removed simulation from cache", simulation_id=simulation_id)
            return True
        else:
            logger.warning(
                "Attempted to remove non-existent simulation from cache",
                simulation_id=simulation_id,
            )
            return False

    async def get_feature_importance(self, session_key: int) -> Dict[str, Any]:
        """
        Get feature importance scores for a specific session.

        Args:
            session_key: Session identifier

        Returns:
            Dictionary mapping feature names to importance scores
        """
        logger.info("Getting feature importance", session_key=session_key)

        # Get session data and fit feature engineering pipeline
        request = DataProcessingRequest(
            session_key=session_key,
            include_weather=True,
            include_grid=True,
            include_lap_times=True,
            include_pit_stops=True,
        )

        processed_response = await self.process_session_data(request)

        # Fit the feature engineering pipeline
        features, targets, metadata = (
            self.feature_engineering_service.fit_transform_features(
                processed_response.processed_data, target_column="lap_time"
            )
        )

        # Get feature importance
        feature_importance = self.feature_engineering_service.get_feature_importance(
            "lap_time"
        )

        return feature_importance  # type: ignore

    async def get_encoding_info(self, session_key: int) -> Dict[str, Any]:
        """
        Get information about categorical encoding for a specific session.

        Args:
            session_key: Session identifier

        Returns:
            Dictionary containing encoding information
        """
        logger.info("Getting encoding information", session_key=session_key)

        # Get session data and fit feature engineering pipeline
        request = DataProcessingRequest(
            session_key=session_key,
            include_weather=True,
            include_grid=True,
            include_lap_times=True,
            include_pit_stops=True,
        )

        processed_response = await self.process_session_data(request)

        # Fit the feature engineering pipeline
        features, targets, metadata = (
            self.feature_engineering_service.fit_transform_features(
                processed_response.processed_data, target_column="lap_time"
            )
        )

        # Get encoding information from feature engineering service
        return self.feature_engineering_service.get_encoding_info()

    async def apply_one_hot_encoding(
        self, request: DataProcessingRequest
    ) -> Dict[str, Any]:
        """
        Apply one-hot encoding to categorical features for a session.

        Args:
            request: Data processing request

        Returns:
            Dictionary containing encoding results
        """
        logger.info("Applying one-hot encoding", session_key=request.session_key)

        start_time = time.time()

        # Process session data
        processed_response = await self.process_session_data(request)

        # Fit the feature engineering pipeline
        features, targets, metadata = (
            self.feature_engineering_service.fit_transform_features(
                processed_response.processed_data, target_column="lap_time"
            )
        )

        # Apply one-hot encoding using feature engineering service
        encoding_result = self.feature_engineering_service.apply_one_hot_encoding(
            self.feature_engineering_service.onehot_columns
        )

        processing_time_ms = int((time.time() - start_time) * 1000)

        # Add processing time to the result
        encoding_result["processing_time_ms"] = processing_time_ms
        encoding_result["features_shape"] = features.shape
        encoding_result["targets_shape"] = targets.shape

        return encoding_result

    async def get_encoding_statistics(self, session_key: int) -> Dict[str, Any]:
        """
        Get encoding statistics for a specific session.

        Args:
            session_key: Session identifier

        Returns:
            Dictionary containing encoding statistics
        """
        logger.info("Getting encoding statistics", session_key=session_key)

        # Get session data and fit feature engineering pipeline
        request = DataProcessingRequest(
            session_key=session_key,
            include_weather=True,
            include_grid=True,
            include_lap_times=True,
            include_pit_stops=True,
        )

        processed_response = await self.process_session_data(request)

        # Fit the feature engineering pipeline
        features, targets, metadata = (
            self.feature_engineering_service.fit_transform_features(
                processed_response.processed_data, target_column="lap_time"
            )
        )

        # Get encoding statistics from feature engineering service
        return self.feature_engineering_service.get_encoding_statistics()

    async def process_session_data(
        self, request: DataProcessingRequest
    ) -> DataProcessingResponse:
        """
        Process and merge all session data into training-ready format.

        Args:
            request: Data processing request with options

        Returns:
            Processed data with merged features
        """
        logger.info("Processing session data", session_key=request.session_key)
        start_time = time.time()

        try:
            async with self.openf1_client as client:
                # Fetch all required data sources
                data_sources = []
                session_info = None
                weather_data = None
                grid_data = None
                lap_times_data = None
                pit_stops_data = None

                # Get session info
                sessions = await client.get_sessions(
                    2024
                )  # TODO: Get year from session_key
                for session in sessions:
                    if session.get("session_key") == request.session_key:
                        session_info = session
                        break

                if not session_info:
                    raise ValueError(f"Session {request.session_key} not found")

                # Fetch weather data if requested
                if request.include_weather:
                    weather_data = await client.get_session_weather_summary(
                        request.session_key
                    )
                    data_sources.append("weather")

                # Fetch grid data if requested
                if request.include_grid:
                    grid_data = await client.get_session_grid_summary(
                        request.session_key
                    )
                    data_sources.append("grid")

                # Fetch lap times data if requested
                if request.include_lap_times:
                    lap_times_data = await client.get_session_lap_times_summary(
                        request.session_key
                    )
                    data_sources.append("lap_times")

                # Fetch pit stops data if requested
                if request.include_pit_stops:
                    pit_stops_data = await client.get_session_pit_stops_summary(
                        request.session_key
                    )
                    data_sources.append("pit_stops")

                # Process and merge the data
                processed_data_points = self._merge_and_process_data(
                    session_info,
                    weather_data,
                    grid_data,
                    lap_times_data,
                    pit_stops_data,
                    request.processing_options or {},
                )

                # Calculate processing statistics
                processing_time_ms = int((time.time() - start_time) * 1000)
                total_data_points = len(processed_data_points)
                total_drivers = len(set(dp.driver_id for dp in processed_data_points))
                total_laps = (
                    max(dp.lap_number for dp in processed_data_points)
                    if processed_data_points
                    else 0
                )

                # Use feature engineering service to process data
                try:
                    features, targets, feature_metadata = (
                        self.feature_engineering_service.fit_transform_features(
                            processed_data_points, target_column="lap_time"
                        )
                    )

                    # Get data quality report
                    data_quality_report = (
                        self.feature_engineering_service.get_data_quality_report(
                            processed_data_points
                        )
                    )

                    # Extract metrics from feature engineering metadata
                    data_quality_score = data_quality_report["data_quality_score"]
                    missing_data_points = sum(
                        report["missing_count"]
                        for report in data_quality_report[
                            "missing_data_summary"
                        ].values()
                    )
                    feature_columns = feature_metadata["feature_columns"]

                    logger.info(
                        "Feature engineering completed successfully",
                        features_shape=features.shape,
                        data_quality_score=data_quality_score,
                    )

                except Exception as e:
                    logger.error("Feature engineering failed", error=str(e))
                    # Fallback to basic metrics
                    missing_data_points = sum(
                        1
                        for dp in processed_data_points
                        if dp.lap_time is None or dp.air_temperature is None
                    )
                    data_quality_score = (
                        1.0 - (missing_data_points / total_data_points)
                        if total_data_points > 0
                        else 0.0
                    )
                    feature_columns = [
                        "lap_number",
                        "tire_compound",
                        "fuel_load",
                        "grid_position",
                        "air_temperature",
                        "track_temperature",
                        "humidity",
                        "weather_condition",
                        "pit_stop_count",
                        "total_pit_time",
                    ]

                target_columns = [
                    "lap_time",
                    "sector_1_time",
                    "sector_2_time",
                    "sector_3_time",
                ]

                # Generate derived features
                features_generated = [
                    "lap_time_normalized",
                    "fuel_load_normalized",
                    "tire_wear_estimate",
                    "position_change",
                    "weather_impact_score",
                ]

                # Create processing summary
                processing_summary = DataProcessingSummary(
                    session_key=request.session_key,
                    total_data_points=total_data_points,
                    total_drivers=total_drivers,
                    total_laps=total_laps,
                    data_sources=data_sources,
                    processing_time_ms=processing_time_ms,
                    missing_data_points=missing_data_points,
                    data_quality_score=data_quality_score,
                    features_generated=features_generated,
                    processing_errors=[],
                )

                return DataProcessingResponse(
                    session_key=request.session_key,
                    session_name=session_info.get("session_name", "Unknown Session"),
                    track_name=session_info.get(
                        "track_name", session_info.get("location", "Unknown Track")
                    ),
                    country=session_info.get(
                        "country_name", session_info.get("country", "Unknown")
                    ),
                    year=session_info.get("year", 2024),
                    processing_summary=processing_summary,
                    processed_data=processed_data_points,
                    feature_columns=feature_columns,
                    target_columns=target_columns,
                    created_at=datetime.now(timezone.utc),
                )

        except Exception as e:
            logger.error(
                "Failed to process session data",
                session_key=request.session_key,
                error=str(e),
                exc_info=True,
            )
            raise

    def _merge_and_process_data(
        self,
        session_info: dict,
        weather_data: Optional[dict],
        grid_data: Optional[dict],
        lap_times_data: Optional[dict],
        pit_stops_data: Optional[dict],
        processing_options: dict,
    ) -> List[ProcessedDataPoint]:
        """
        Merge and process data from multiple sources.

        Args:
            session_info: Session information
            weather_data: Weather data summary
            grid_data: Grid data summary
            lap_times_data: Lap times data
            pit_stops_data: Pit stops data
            processing_options: Processing options

        Returns:
            List of processed data points
        """
        processed_points: List[ProcessedDataPoint] = []

        if not lap_times_data or not lap_times_data.get("lap_times"):
            return processed_points

        # Create mappings for efficient lookup
        grid_positions: Dict[int, int] = {}
        if grid_data and grid_data.get("grid_positions"):
            for pos in grid_data["grid_positions"]:
                driver_id = pos.get("driver_id")
                if driver_id:
                    grid_positions[driver_id] = pos.get("position")

        pit_stops_by_driver: Dict[int, List[Dict]] = {}
        if pit_stops_data and pit_stops_data.get("pit_stops"):
            for pit in pit_stops_data["pit_stops"]:
                driver_id = pit.get("driver_id")
                if driver_id is not None:
                    if driver_id not in pit_stops_by_driver:
                        pit_stops_by_driver[driver_id] = []
                    pit_stops_by_driver[driver_id].append(pit)

        # Process each lap time
        for lap_data in lap_times_data.get("lap_times", []):
            driver_id = lap_data.get("driver_id")
            lap_number = lap_data.get("lap_number")

            if not driver_id or not lap_number:
                continue

            # Calculate pit stop metrics for this driver up to this lap
            pit_stop_count = 0
            total_pit_time = 0.0
            driver_pits = pit_stops_by_driver.get(driver_id, [])
            for pit in driver_pits:
                if pit.get("lap_number", 0) <= lap_number:
                    pit_stop_count += 1
                    total_pit_time += pit.get("pit_duration", 0.0)

            # Create processed data point
            processed_point = ProcessedDataPoint(
                timestamp=datetime.fromisoformat(
                    lap_data.get("timestamp", "").replace("Z", "+00:00")
                ),
                driver_id=driver_id,
                lap_number=lap_number,
                lap_time=lap_data.get("lap_time"),
                sector_1_time=lap_data.get("sector_1_time"),
                sector_2_time=lap_data.get("sector_2_time"),
                sector_3_time=lap_data.get("sector_3_time"),
                tire_compound=lap_data.get("tire_compound"),
                fuel_load=lap_data.get("fuel_load"),
                grid_position=grid_positions.get(driver_id),
                current_position=1,  # TODO: Calculate from race data
                air_temperature=(
                    weather_data.get("avg_air_temperature") if weather_data else None
                ),
                track_temperature=(
                    weather_data.get("avg_track_temperature") if weather_data else None
                ),
                humidity=weather_data.get("avg_humidity") if weather_data else None,
                weather_condition=(
                    weather_data.get("weather_condition") if weather_data else "unknown"
                ),
                track_type=None,  # TODO: Get from track data
                driver_team=None,  # TODO: Get from driver data
                pit_stop_count=pit_stop_count,
                total_pit_time=total_pit_time,
                lap_status=lap_data.get("lap_status", "valid"),
            )

            processed_points.append(processed_point)

        # Sort by timestamp and lap number
        processed_points.sort(key=lambda x: (x.timestamp, x.lap_number))

        return processed_points

    def _prepare_features(
        self, request: SimulationRequest, historical_data: dict
    ) -> List[float]:
        """
        Prepare features for ML model prediction.

        The trained model expects 4 features: [lap_number, driver_number, i2_speed, st_speed]
        We'll map available simulation data to these expected features.

        Args:
            request: Simulation request
            historical_data: Historical performance data from OpenF1Client

        Returns:
            Feature vector for prediction (4 features)
        """
        # Map simulation data to trained model's expected features
        # Our trained model expects: [lap_number, driver_number, i2_speed, st_speed]

        # Use historical data to prepare features
        # 'data_points' from the client represents the number of historical laps found.
        lap_number = historical_data.get("data_points", 50) // 2  # Assume mid-race
        driver_number = request.driver_id

        # TODO: FWI-BE-111 - Enhance OpenF1Client to fetch speed trap data.
        # For now, using default values as the current client does not provide them.
        i2_speed = 240.0
        st_speed = 300.0

        features = [
            lap_number,  # lap_number
            driver_number,  # driver_number
            i2_speed,  # i2_speed
            st_speed,  # st_speed
        ]

        return features

    def _calculate_confidence_score(self, historical_data: dict) -> float:
        """
        Calculate confidence score based on data quality.

        Args:
            historical_data: Historical performance data

        Returns:
            Confidence score between 0 and 1
        """
        # Simple confidence calculation based on data availability
        data_points = historical_data.get("data_points", 0)
        if data_points >= 50:
            return 0.95
        elif data_points >= 20:
            return 0.85
        elif data_points >= 10:
            return 0.75
        elif data_points >= 5:
            return 0.60
        else:
            return 0.40

    async def _get_historical_data(
        self, driver_id: int, track_id: int, season: int
    ) -> dict:
        """
        Get historical data for a driver and track using the OpenF1Client.

        Args:
            driver_id: Driver identifier
            track_id: Track identifier
            season: F1 season year

        Returns:
            Historical performance data
        """
        logger.info(
            "Fetching historical data from OpenF1Client",
            driver_id=driver_id,
            track_id=track_id,
            season=season,
        )
        async with self.openf1_client as client:
            return await client.get_historical_data(  # type: ignore
                driver_id=driver_id, track_id=track_id, season=season
            )

    # FWI-BE-106: Enhanced Categorical Encoding Methods

    async def encode_categorical_feature(
        self, request: CategoricalEncodingRequest
    ) -> CategoricalMappingResponse:
        """
        Encode a specific categorical feature with enhanced mapping and validation.

        Args:
            request: Categorical encoding request

        Returns:
            Categorical mapping response with detailed encoding information
        """
        from datetime import datetime

        logger.info(
            "Encoding categorical feature",
            session_key=request.session_key,
            feature_name=request.feature_name,
            encoding_type=request.encoding_type,
        )

        # Process session data to get the categorical feature
        data_request = DataProcessingRequest(session_key=request.session_key)
        processed_response = await self.process_session_data(data_request)

        # Fit the feature engineering pipeline
        features, targets, metadata = (
            self.feature_engineering_service.fit_transform_features(
                processed_response.processed_data, target_column="lap_time"
            )
        )

        # Get the specific encoder for the requested feature
        encoding_result = self.feature_engineering_service.encode_categorical_feature(
            request.feature_name, request.encoding_type
        )

        categories = encoding_result["categories"]
        feature_mappings = encoding_result["feature_mappings"]
        encoded_feature_names = encoding_result["encoded_feature_names"]
        validation_passed = encoding_result["validation_passed"]

        # Create encoding metadata
        encoding_metadata = {
            "feature_count": len(encoded_feature_names),
            "sparse_encoding": False,
            "handle_unknown": "ignore",
            "encoding_version": "1.0",
            "encoding_timestamp": datetime.now(timezone.utc).isoformat(),
        }

        return CategoricalMappingResponse(
            session_key=request.session_key,
            feature_name=request.feature_name,
            encoding_type=request.encoding_type,
            categories=categories,
            feature_mappings=feature_mappings,
            encoded_feature_names=encoded_feature_names,
            encoding_metadata=encoding_metadata,
            validation_passed=validation_passed,
            created_at=datetime.now(timezone.utc),
        )

    async def validate_categorical_encodings(
        self, request: DataProcessingRequest
    ) -> EncodingValidationResponse:
        """
        Validate consistency of all categorical encodings for a session.

        Args:
            request: Data processing request

        Returns:
            Encoding validation response with detailed validation results
        """
        logger.info("Validating categorical encodings", session_key=request.session_key)

        start_time = time.time()

        # Process session data
        processed_response = await self.process_session_data(request)

        # Fit the feature engineering pipeline
        features, targets, metadata = (
            self.feature_engineering_service.fit_transform_features(
                processed_response.processed_data, target_column="lap_time"
            )
        )

        # Use the feature engineering service to validate encodings
        validation_result = (
            self.feature_engineering_service.validate_categorical_encodings()
        )

        feature_validations = validation_result["feature_validations"]
        encoding_consistency = validation_result["encoding_consistency"]
        validation_errors = validation_result["validation_errors"]

        # Overall validation status
        validation_passed = validation_result["validation_passed"]

        processing_time_ms = int((time.time() - start_time) * 1000)

        return EncodingValidationResponse(
            session_key=request.session_key,
            total_features_validated=validation_result["total_features_validated"],
            validation_passed=validation_passed,
            feature_validations=feature_validations,
            encoding_consistency=encoding_consistency,
            validation_errors=validation_errors,
            validation_time_ms=processing_time_ms,
        )
