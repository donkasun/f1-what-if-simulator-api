"""
Simulation service for F1 What-If Simulator business logic.
"""

import time
import uuid
from datetime import datetime
from typing import List

import structlog
from async_lru import alru_cache

from app.api.v1.schemas import (
    DriverResponse,
    SimulationRequest,
    SimulationResponse,
    TrackResponse,
)
from app.core.exceptions import (
    DriverNotFoundError,
    InvalidSimulationParametersError,
    OpenF1APIError,
)
from app.external.openf1_client import OpenF1Client
from app.models.model_loader import ModelLoader

logger = structlog.get_logger()


class SimulationService:
    """Service class for handling F1 simulation business logic."""
    
    def __init__(self):
        """Initialize the simulation service with dependencies."""
        self.openf1_client = OpenF1Client()
        self.model_loader = ModelLoader()
        self._simulation_cache = {}  # In-memory cache for simulation results
    
    @alru_cache(maxsize=100)
    async def get_drivers(self, season: int) -> List[DriverResponse]:
        """
        Get all drivers for a specific season.
        
        Args:
            season: F1 season year
            
        Returns:
            List of driver information
            
        Raises:
            OpenF1APIError: If external API call fails
        """
        logger.info("Fetching drivers from external API", season=season)
        
        try:
            drivers_data = await self.openf1_client.get_drivers(season)
            
            drivers = []
            for driver_data in drivers_data:
                driver = DriverResponse(
                    driver_id=driver_data["driver_id"],
                    name=driver_data["name"],
                    code=driver_data["code"],
                    team=driver_data["team"],
                    nationality=driver_data["nationality"]
                )
                drivers.append(driver)
            
            logger.info("Successfully fetched drivers", season=season, count=len(drivers))
            return drivers
            
        except Exception as e:
            logger.error("Failed to fetch drivers", season=season, error=str(e), exc_info=True)
            raise OpenF1APIError(f"Failed to fetch drivers for season {season}", 500)
    
    @alru_cache(maxsize=50)
    async def get_tracks(self, season: int) -> List[TrackResponse]:
        """
        Get all tracks for a specific season.
        
        Args:
            season: F1 season year
            
        Returns:
            List of track information
            
        Raises:
            OpenF1APIError: If external API call fails
        """
        logger.info("Fetching tracks from external API", season=season)
        
        try:
            tracks_data = await self.openf1_client.get_tracks(season)
            
            tracks = []
            for track_data in tracks_data:
                track = TrackResponse(
                    track_id=track_data["track_id"],
                    name=track_data["name"],
                    country=track_data["country"],
                    circuit_length=track_data["circuit_length"],
                    number_of_laps=track_data["number_of_laps"]
                )
                tracks.append(track)
            
            logger.info("Successfully fetched tracks", season=season, count=len(tracks))
            return tracks
            
        except Exception as e:
            logger.error("Failed to fetch tracks", season=season, error=str(e), exc_info=True)
            raise OpenF1APIError(f"Failed to fetch tracks for season {season}", 500)
    
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
                raise InvalidSimulationParametersError(f"Track with ID {request.track_id} not found")
            
            # Get historical data for the driver and track
            historical_data = await self.openf1_client.get_historical_data(
                driver_id=request.driver_id,
                track_id=request.track_id,
                season=request.season
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
                created_at=datetime.utcnow(),
                processing_time_ms=processing_time_ms
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
            raise InvalidSimulationParametersError(f"Simulation {simulation_id} not found")
        
        return self._simulation_cache[simulation_id]
    
    def _prepare_features(self, request: SimulationRequest, historical_data: dict) -> List[float]:
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