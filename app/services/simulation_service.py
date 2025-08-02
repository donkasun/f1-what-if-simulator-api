"""
Simulation service for F1 What-If Simulator.

This module provides business logic for running F1 simulations,
including ML model integration and result caching.
"""

import time
import uuid
from datetime import datetime, UTC
from typing import List

import structlog
from async_lru import alru_cache

from app.core.exceptions import DriverNotFoundError, InvalidSimulationParametersError
from app.external.openf1_client import OpenF1Client
from app.models.model_loader import ModelLoader
from app.api.v1.schemas import (
    DriverResponse,
    SimulationRequest,
    SimulationResponse,
    TrackResponse,
)

logger = structlog.get_logger()

# Global cache for simulation results - persists across service instances
_simulation_cache = {}


class SimulationService:
    """Service class for handling F1 simulation business logic."""

    def __init__(self):
        """Initialize the simulation service with dependencies."""
        self.openf1_client = OpenF1Client()
        self.model_loader = ModelLoader()
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
        logger.info("Fetching mock drivers for season", season=season)

        # Mock driver data for development
        mock_drivers = [
            DriverResponse(
                driver_id=1,
                name="Max Verstappen",
                code="VER",
                team="Red Bull Racing",
                nationality="Dutch",
            ),
            DriverResponse(
                driver_id=2,
                name="Lewis Hamilton",
                code="HAM",
                team="Mercedes",
                nationality="British",
            ),
            DriverResponse(
                driver_id=3,
                name="Charles Leclerc",
                code="LEC",
                team="Ferrari",
                nationality="Monegasque",
            ),
            DriverResponse(
                driver_id=4,
                name="Lando Norris",
                code="NOR",
                team="McLaren",
                nationality="British",
            ),
            DriverResponse(
                driver_id=5,
                name="Carlos Sainz",
                code="SAI",
                team="Ferrari",
                nationality="Spanish",
            ),
        ]

        logger.info(
            "Successfully fetched mock drivers", season=season, count=len(mock_drivers)
        )
        return mock_drivers

    @alru_cache(maxsize=50)
    async def get_tracks(self, season: int) -> List[TrackResponse]:
        """
        Get all tracks for a specific season.

        Args:
            season: F1 season year

        Returns:
            List of track information
        """
        logger.info("Fetching mock tracks for season", season=season)

        # Mock track data for development
        mock_tracks = [
            TrackResponse(
                track_id=1,
                name="Monaco Grand Prix",
                country="Monaco",
                circuit_length=3.337,
                number_of_laps=78,
            ),
            TrackResponse(
                track_id=2,
                name="Silverstone Circuit",
                country="United Kingdom",
                circuit_length=5.891,
                number_of_laps=52,
            ),
            TrackResponse(
                track_id=3,
                name="Spa-Francorchamps",
                country="Belgium",
                circuit_length=7.004,
                number_of_laps=44,
            ),
            TrackResponse(
                track_id=4,
                name="Monza",
                country="Italy",
                circuit_length=5.793,
                number_of_laps=53,
            ),
            TrackResponse(
                track_id=5,
                name="Suzuka",
                country="Japan",
                circuit_length=5.807,
                number_of_laps=53,
            ),
        ]

        logger.info(
            "Successfully fetched mock tracks", season=season, count=len(mock_tracks)
        )
        return mock_tracks

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

            # Get mock historical data for the driver and track
            historical_data = self._get_mock_historical_data(
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
                weather_conditions=request.weather_conditions,
                car_setup=request.car_setup,
                created_at=datetime.now(UTC),
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

            return result

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

        return self._simulation_cache[simulation_id]

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
        Remove a specific simulation from cache.

        Args:
            simulation_id: Simulation ID to remove

        Returns:
            True if removed, False if not found
        """
        if simulation_id in self._simulation_cache:
            del self._simulation_cache[simulation_id]
            logger.info("Removed simulation from cache", simulation_id=simulation_id)
            return True
        return False

    def _prepare_features(
        self, request: SimulationRequest, historical_data: dict
    ) -> List[float]:
        """
        Prepare features for ML model prediction.

        Args:
            request: Simulation request
            historical_data: Historical performance data

        Returns:
            Feature vector for prediction
        """
        # This is a simplified feature preparation
        # In a real implementation, this would be much more sophisticated
        features = [
            request.driver_id,
            request.track_id,
            request.season,
            historical_data.get("avg_lap_time", 0.0),
            historical_data.get("best_lap_time", 0.0),
            historical_data.get("consistency_score", 0.0),
            1.0 if request.weather_conditions == "dry" else 0.0,
            1.0 if request.weather_conditions == "wet" else 0.0,
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

    def _get_mock_historical_data(
        self, driver_id: int, track_id: int, season: int
    ) -> dict:
        """
        Get mock historical data for development purposes.

        Args:
            driver_id: Driver identifier
            track_id: Track identifier
            season: F1 season year

        Returns:
            Mock historical performance data
        """
        # Generate realistic mock data based on driver and track
        base_lap_time = 75.0 + (driver_id * 0.5) + (track_id * 1.2)

        return {
            "avg_lap_time": base_lap_time,
            "best_lap_time": base_lap_time - 2.0,
            "consistency_score": 0.85 + (driver_id * 0.02),
            "data_points": 25 + (driver_id * 5),
            "last_race_position": max(1, driver_id),
            "qualifying_position": max(1, driver_id),
            "weather_conditions": ["dry", "wet", "intermediate"],
            "tire_usage": {"soft": 0.3, "medium": 0.4, "hard": 0.3},
        }
