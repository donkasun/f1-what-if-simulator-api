"""
Tests for API endpoints.
"""

import pytest
import numpy as np
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch, Mock, MagicMock
from app.main import app
from app.core.exceptions import (
    InvalidSimulationParametersError,
    FeatureEngineeringError,
)
from app.api.v1.schemas import (
    SimulationResponse,
    StartingGridResponse,
    GridPositionResponse,
    GridSummaryResponse,
    SimulationRequest,
)
from datetime import datetime, UTC


client = TestClient(app)


class TestMainApplication:
    """Test cases for main application."""

    def test_create_app(self):
        """Test create_app function."""
        from app.main import create_app

        app = create_app()
        assert app is not None
        assert app.title == "F1 What-If Simulator API"
        assert app.version == "0.1.0"

    def test_lifespan_manager(self):
        """Test lifespan manager."""
        from app.main import lifespan
        from fastapi import FastAPI

        app = FastAPI()

        # Test that the lifespan manager can be used as a context manager
        async def test_lifespan():
            async with lifespan(app):
                pass

        # This should not raise an exception
        import asyncio

        asyncio.run(test_lifespan())

    def test_exception_handlers(self):
        """Test exception handlers."""
        from app.core.exceptions import (
            DriverNotFoundError,
            InvalidSimulationParametersError,
            OpenF1APIError,
        )

        # Test DriverNotFoundError handler
        with pytest.raises(DriverNotFoundError):
            raise DriverNotFoundError(123)

        # Test InvalidSimulationParametersError handler
        with pytest.raises(InvalidSimulationParametersError):
            raise InvalidSimulationParametersError("Test error")

        # Test OpenF1APIError handler
        with pytest.raises(OpenF1APIError):
            raise OpenF1APIError("Test error", 500)


class TestModelLoader:
    """Test cases for model loader."""

    @pytest.mark.asyncio
    async def test_model_loader_initialization(self):
        """Test ModelLoader initialization."""
        from app.models.model_loader import ModelLoader

        loader = ModelLoader()
        assert loader._model is None
        assert loader._model_path is not None

    @pytest.mark.asyncio
    async def test_get_model_info_not_loaded(self):
        """Test get_model_info when model is not loaded."""
        from app.models.model_loader import ModelLoader

        loader = ModelLoader()
        info = loader.get_model_info()
        assert info["status"] == "not_loaded"

    @pytest.mark.asyncio
    async def test_reload_model(self):
        """Test reload_model method."""
        from app.models.model_loader import ModelLoader

        loader = ModelLoader()
        loader.reload_model()
        # Should not raise an exception

    @pytest.mark.asyncio
    async def test_create_dummy_model(self):
        """Test _create_dummy_model method."""
        from app.models.model_loader import ModelLoader

        loader = ModelLoader()
        model = loader._create_dummy_model()
        assert model is not None
        assert hasattr(model, "predict")

    @pytest.mark.asyncio
    async def test_get_model_info_loaded(self):
        """Test get_model_info when model is loaded."""
        from app.models.model_loader import ModelLoader

        loader = ModelLoader()
        # Load a dummy model first
        loader._model = loader._create_dummy_model()

        info = loader.get_model_info()
        assert info["status"] == "loaded"
        assert "model_path" in info
        assert "model_type" in info
        assert "n_estimators" in info


class TestLoggingConfiguration:
    """Test cases for logging configuration."""

    def test_get_logger(self):
        """Test get_logger function."""
        from app.core.logging_config import get_logger

        logger = get_logger("test_logger")
        assert logger is not None
        assert hasattr(logger, "info")
        assert hasattr(logger, "error")
        assert hasattr(logger, "warning")

    def test_log_request_info(self):
        """Test log_request_info function."""
        from app.core.logging_config import log_request_info, get_logger

        logger = get_logger("test_request")
        request_info = {
            "method": "GET",
            "url": "/api/v1/health",
            "client_ip": "127.0.0.1",
            "user_agent": "test-agent",
            "request_id": "req-123",
        }

        # This should not raise an exception
        log_request_info(logger, request_info)

    def test_log_response_info(self):
        """Test log_response_info function."""
        from app.core.logging_config import log_response_info, get_logger

        logger = get_logger("test_response")
        response_info = {
            "status_code": 200,
            "response_time_ms": 150,
            "request_id": "req-123",
        }

        # This should not raise an exception
        log_response_info(logger, response_info)

    def test_log_external_api_call_success(self):
        """Test log_external_api_call function for successful calls."""
        from app.core.logging_config import log_external_api_call, get_logger

        logger = get_logger("test_api")

        # This should not raise an exception
        log_external_api_call(
            logger=logger,
            api_name="OpenF1",
            endpoint="/drivers",
            method="GET",
            status_code=200,
            response_time_ms=100,
        )

    def test_log_external_api_call_error(self):
        """Test log_external_api_call function for failed calls."""
        from app.core.logging_config import log_external_api_call, get_logger

        logger = get_logger("test_api")

        # This should not raise an exception
        log_external_api_call(
            logger=logger,
            api_name="OpenF1",
            endpoint="/drivers",
            method="GET",
            status_code=500,
            response_time_ms=100,
            error="Connection timeout",
        )

    def test_log_simulation_event(self):
        """Test log_simulation_event function."""
        from app.core.logging_config import log_simulation_event, get_logger

        logger = get_logger("test_simulation")

        # This should not raise an exception
        log_simulation_event(
            logger=logger,
            event_type="started",
            simulation_id="sim-123",
            driver_id=1,
            track_id=1,
            season=2024,
            weather_conditions="dry",
        )


class TestCustomExceptions:
    """Test cases for custom exceptions."""

    def test_f1_simulator_error(self):
        """Test F1SimulatorError base exception."""
        from app.core.exceptions import F1SimulatorError

        error = F1SimulatorError("Test error", "TEST_ERROR")
        assert error.message == "Test error"
        assert error.code == "TEST_ERROR"
        assert str(error) == "Test error"

    def test_driver_not_found_error(self):
        """Test DriverNotFoundError exception."""
        from app.core.exceptions import DriverNotFoundError

        error = DriverNotFoundError(123)
        assert error.driver_id == 123
        assert error.message == "Driver with ID 123 not found"
        assert error.code == "DRIVER_NOT_FOUND"

    def test_track_not_found_error(self):
        """Test TrackNotFoundError exception."""
        from app.core.exceptions import TrackNotFoundError

        error = TrackNotFoundError(456)
        assert error.track_id == 456
        assert error.message == "Track with ID 456 not found"
        assert error.code == "TRACK_NOT_FOUND"

    def test_invalid_simulation_parameters_error(self):
        """Test InvalidSimulationParametersError exception."""
        from app.core.exceptions import InvalidSimulationParametersError

        error = InvalidSimulationParametersError("Invalid parameters")
        assert error.details == "Invalid parameters"
        assert error.message == "Invalid simulation parameters: Invalid parameters"
        assert error.code == "INVALID_SIMULATION_PARAMETERS"

    def test_model_load_error(self):
        """Test ModelLoadError exception."""
        from app.core.exceptions import ModelLoadError

        error = ModelLoadError("/path/to/model", "File not found")
        assert error.model_path == "/path/to/model"
        assert error.error == "File not found"
        assert (
            error.message == "Failed to load model from /path/to/model: File not found"
        )
        assert error.code == "MODEL_LOAD_ERROR"

    def test_openf1_api_error(self):
        """Test OpenF1APIError exception."""
        from app.core.exceptions import OpenF1APIError

        error = OpenF1APIError("API request failed", 500)
        assert error.status_code == 500
        assert error.message == "API request failed"
        assert error.code == "OPENF1_API_ERROR"

    def test_simulation_execution_error(self):
        """Test SimulationExecutionError exception."""
        from app.core.exceptions import SimulationExecutionError

        error = SimulationExecutionError("sim_123", "Execution failed")
        assert error.simulation_id == "sim_123"
        assert error.error == "Execution failed"
        assert error.message == "Simulation sim_123 failed: Execution failed"
        assert error.code == "SIMULATION_EXECUTION_ERROR"

    def test_cache_error(self):
        """Test CacheError exception."""
        from app.core.exceptions import CacheError

        error = CacheError("get", "Key not found")
        assert error.operation == "get"
        assert error.error == "Key not found"
        assert error.message == "Cache get failed: Key not found"
        assert error.code == "CACHE_ERROR"

    def test_feature_engineering_error(self):
        """Test FeatureEngineeringError exception."""
        from app.core.exceptions import FeatureEngineeringError

        error = FeatureEngineeringError("Feature engineering failed")
        assert error.message == "Feature engineering failed"
        assert error.code == "FEATURE_ENGINEERING_ERROR"


class TestDriversEndpoint:
    """Test cases for /api/v1/drivers endpoint."""

    @pytest.mark.asyncio
    async def test_get_drivers_success(self):
        """Test successful retrieval of drivers."""
        mock_drivers = [
            {
                "driver_id": 1,
                "name": "Max Verstappen",
                "code": "VER",
                "team": "Red Bull Racing",
                "nationality": "NED",
            }
        ]

        with patch("app.api.v1.endpoints.SimulationService") as mock_service_class:
            mock_service = mock_service_class.return_value
            mock_service.get_drivers = AsyncMock(return_value=mock_drivers)

            response = client.get("/api/v1/drivers?season=2024")

            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, list)
            assert len(data) == 1
            assert data[0]["driver_id"] == 1
            assert data[0]["name"] == "Max Verstappen"

    @pytest.mark.asyncio
    async def test_get_drivers_service_error(self):
        """Test drivers endpoint when service raises an error."""
        with patch("app.api.v1.endpoints.SimulationService") as mock_service_class:
            mock_service = mock_service_class.return_value
            mock_service.get_drivers = AsyncMock(side_effect=Exception("Service error"))

            response = client.get("/api/v1/drivers?season=2024")

            assert response.status_code == 500
            assert "Failed to fetch drivers" in response.json()["detail"]


class TestTracksEndpoint:
    """Test cases for /api/v1/tracks endpoint."""

    @pytest.mark.asyncio
    async def test_get_tracks_success(self):
        """Test successful retrieval of tracks."""
        mock_tracks = [
            {
                "track_id": 1,
                "name": "Monaco",
                "country": "Monaco",
                "circuit_length": 3.337,
            }
        ]

        with patch("app.api.v1.endpoints.SimulationService") as mock_service_class:
            mock_service = mock_service_class.return_value
            mock_service.get_tracks = AsyncMock(return_value=mock_tracks)

            response = client.get("/api/v1/tracks?season=2024")

            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, list)
            assert len(data) == 1
            assert data[0]["track_id"] == 1
            assert data[0]["name"] == "Monaco"

    @pytest.mark.asyncio
    async def test_get_tracks_service_error(self):
        """Test tracks endpoint when service raises an error."""
        with patch("app.api.v1.endpoints.SimulationService") as mock_service_class:
            mock_service = mock_service_class.return_value
            mock_service.get_tracks = AsyncMock(side_effect=Exception("Service error"))

            response = client.get("/api/v1/tracks?season=2024")

            assert response.status_code == 500
            assert "Failed to fetch tracks" in response.json()["detail"]


class TestSessionsEndpoint:
    """Test cases for /api/v1/sessions endpoint."""

    @pytest.mark.asyncio
    async def test_get_sessions_success(self):
        """Test successful retrieval of sessions."""
        mock_sessions = [
            {
                "session_key": 12345,
                "meeting_key": 67890,
                "location": "Monaco",
                "session_type": "Race",
                "session_name": "Monaco Grand Prix",
                "date_start": "2024-05-26T14:00:00Z",
                "date_end": "2024-05-26T16:00:00Z",
                "country_name": "Monaco",
                "circuit_short_name": "Monaco",
                "year": 2024,
            }
        ]

        with patch("app.api.v1.endpoints.SimulationService") as mock_service_class:
            mock_service = mock_service_class.return_value
            mock_service.get_sessions = AsyncMock(return_value=mock_sessions)

            response = client.get("/api/v1/sessions?season=2024")

            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, list)
            assert len(data) == 1
            assert data[0]["session_key"] == 12345
            assert data[0]["location"] == "Monaco"

    @pytest.mark.asyncio
    async def test_get_sessions_invalid_season(self):
        """Test sessions endpoint with invalid season."""
        response = client.get("/api/v1/sessions?season=invalid")

        assert response.status_code == 422  # Validation error

    @pytest.mark.asyncio
    async def test_get_sessions_missing_season(self):
        """Test sessions endpoint without season parameter."""
        response = client.get("/api/v1/sessions")

        assert response.status_code == 422  # Validation error

    @pytest.mark.asyncio
    async def test_get_sessions_service_error(self):
        """Test sessions endpoint when service raises an error."""
        with patch("app.api.v1.endpoints.SimulationService") as mock_service_class:
            mock_service = mock_service_class.return_value
            mock_service.get_sessions = AsyncMock(
                side_effect=Exception("Service error")
            )

            response = client.get("/api/v1/sessions?season=2024")

            assert response.status_code == 500


class TestWeatherDataEndpoint:
    """Test cases for /api/v1/weather/{session_key} endpoint."""

    @pytest.mark.asyncio
    async def test_get_weather_data_success(self):
        """Test successful retrieval of weather data."""
        mock_weather_data = [
            {
                "date": "2024-05-26T14:00:00Z",
                "session_key": 12345,
                "air_temperature": 25.5,
                "track_temperature": 35.2,
                "humidity": 65.0,
                "pressure": 1013.25,
                "wind_speed": 5.2,
                "wind_direction": 180,
                "rainfall": 0.0,
            }
        ]

        with patch("app.api.v1.endpoints.SimulationService") as mock_service_class:
            mock_service = mock_service_class.return_value
            mock_service.get_weather_data = AsyncMock(return_value=mock_weather_data)

            response = client.get("/api/v1/weather/12345")

            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, list)
            assert len(data) == 1
            assert data[0]["session_key"] == 12345
            assert data[0]["air_temperature"] == 25.5

    @pytest.mark.asyncio
    async def test_get_weather_data_invalid_session_key(self):
        """Test weather data endpoint with invalid session key."""
        response = client.get("/api/v1/weather/invalid")

        assert response.status_code == 422  # Validation error

    @pytest.mark.asyncio
    async def test_get_weather_data_not_found(self):
        """Test weather data endpoint when session not found."""
        with patch("app.api.v1.endpoints.SimulationService") as mock_service_class:
            mock_service = mock_service_class.return_value
            mock_service.get_weather_data = AsyncMock(return_value=[])

            response = client.get("/api/v1/weather/99999")

            assert response.status_code == 200
            data = response.json()
            assert data == []

    @pytest.mark.asyncio
    async def test_get_weather_data_service_error(self):
        """Test weather data endpoint when service raises an error."""
        with patch("app.api.v1.endpoints.SimulationService") as mock_service_class:
            mock_service = mock_service_class.return_value
            mock_service.get_weather_data = AsyncMock(
                side_effect=Exception("Service error")
            )

            response = client.get("/api/v1/weather/12345")

            assert response.status_code == 500


class TestWeatherSummaryEndpoint:
    """Test cases for /api/v1/weather/{session_key}/summary endpoint."""

    @pytest.mark.asyncio
    async def test_get_weather_summary_success(self):
        """Test successful retrieval of weather summary."""
        mock_summary = {
            "session_key": 12345,
            "weather_condition": "dry",
            "avg_air_temperature": 25.5,
            "avg_track_temperature": 35.2,
            "avg_humidity": 65.0,
            "avg_pressure": 1013.25,
            "avg_wind_speed": 5.2,
            "total_rainfall": 0.0,
            "data_points": 10,
        }

        with patch("app.api.v1.endpoints.SimulationService") as mock_service_class:
            mock_service = mock_service_class.return_value
            mock_service.get_session_weather_summary = AsyncMock(
                return_value=mock_summary
            )

            response = client.get("/api/v1/weather/12345/summary")

            assert response.status_code == 200
            data = response.json()
            assert data["session_key"] == 12345
            assert data["weather_condition"] == "dry"
            assert data["avg_air_temperature"] == 25.5
            assert data["data_points"] == 10

    @pytest.mark.asyncio
    async def test_get_weather_summary_invalid_session_key(self):
        """Test weather summary endpoint with invalid session key."""
        response = client.get("/api/v1/weather/invalid/summary")

        assert response.status_code == 422  # Validation error

    @pytest.mark.asyncio
    async def test_get_weather_summary_service_error(self):
        """Test weather summary endpoint when service raises an error."""
        with patch("app.api.v1.endpoints.SimulationService") as mock_service_class:
            mock_service = mock_service_class.return_value
            mock_service.get_session_weather_summary = AsyncMock(
                side_effect=Exception("Service error")
            )

            response = client.get("/api/v1/weather/12345/summary")

            assert response.status_code == 500


class TestSimulationEndpoints:
    """Test cases for simulation endpoints."""

    @pytest.mark.asyncio
    async def test_run_simulation_success(self):
        """Test successful simulation execution."""
        mock_result = SimulationResponse(
            simulation_id="sim_123",
            driver_id=1,
            track_id=1,
            season=2024,
            predicted_lap_time=85.234,
            confidence_score=0.92,
            weather_conditions="dry",
            car_setup={},
            created_at=datetime.now(UTC),
            processing_time_ms=1000,
        )

        with patch("app.api.v1.endpoints.SimulationService") as mock_service_class:
            mock_service = mock_service_class.return_value
            mock_service.run_simulation = AsyncMock(return_value=mock_result)

            simulation_request = {
                "season": 2024,
                "driver_id": 1,
                "track_id": 1,
                "weather_conditions": "dry",
                "starting_position": 1,
                "car_setup": {},
            }

            response = client.post("/api/v1/simulate", json=simulation_request)

            assert response.status_code == 200
            data = response.json()
            assert data["simulation_id"] == "sim_123"
            assert data["predicted_lap_time"] == 85.234

    @pytest.mark.asyncio
    async def test_run_simulation_invalid_request(self):
        """Test simulation with invalid request data."""
        invalid_request = {
            "season": "invalid",
            "driver_id": "not_a_number",
            "track_id": 1,
            "weather_conditions": "dry",
            "starting_position": 1,
            "car_setup": {},
        }

        response = client.post("/api/v1/simulate", json=invalid_request)

        assert response.status_code == 422  # Validation error

    @pytest.mark.asyncio
    async def test_run_simulation_service_error(self):
        """Test simulation endpoint when service raises an error."""
        request_data = {
            "driver_id": 1,
            "track_id": 1,
            "season": 2024,
            "weather_conditions": "dry",
            "car_setup": {},
        }

        with patch("app.api.v1.endpoints.SimulationService") as mock_service_class:
            mock_service = mock_service_class.return_value
            mock_service.run_simulation = AsyncMock(
                side_effect=Exception("Simulation failed")
            )

            response = client.post("/api/v1/simulate", json=request_data)

            assert response.status_code == 500
            assert "Simulation failed" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_get_simulation_result_success(self):
        """Test successful retrieval of simulation result."""
        mock_result = SimulationResponse(
            simulation_id="sim_123",
            driver_id=1,
            track_id=1,
            season=2024,
            predicted_lap_time=85.234,
            confidence_score=0.92,
            weather_conditions="dry",
            car_setup={},
            created_at=datetime.now(UTC),
            processing_time_ms=1000,
        )

        with patch("app.api.v1.endpoints.SimulationService") as mock_service_class:
            mock_service = mock_service_class.return_value
            mock_service.get_simulation_result = AsyncMock(return_value=mock_result)

            response = client.get("/api/v1/simulation/sim_123")

            assert response.status_code == 200
            data = response.json()
            assert data["simulation_id"] == "sim_123"

    @pytest.mark.asyncio
    async def test_get_simulation_result_not_found(self):
        """Test simulation result retrieval when not found."""
        with patch("app.api.v1.endpoints.SimulationService") as mock_service_class:
            mock_service = mock_service_class.return_value
            mock_service.get_simulation_result = AsyncMock(
                side_effect=InvalidSimulationParametersError("Simulation not found")
            )

            response = client.get("/api/v1/simulation/nonexistent")

            assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_get_simulation_result_service_error(self):
        """Test get simulation result when service raises an error."""
        with patch("app.api.v1.endpoints.SimulationService") as mock_service_class:
            mock_service = mock_service_class.return_value
            mock_service.get_simulation_result = AsyncMock(
                side_effect=Exception("Service error")
            )

            response = client.get("/api/v1/simulation/test-id")

            assert response.status_code == 500
            assert "Failed to fetch simulation result" in response.json()["detail"]


class TestCacheEndpoints:
    """Test cases for cache management endpoints."""

    @patch("app.api.v1.endpoints.SimulationService")
    def test_get_cache_stats(self, mock_service_class):
        """Test getting cache statistics."""
        mock_stats = {
            "cache_size": 5,
            "cached_simulations": ["sim1", "sim2"],
            "cache_hits": 10,
            "cache_misses": 2,
        }
        mock_service = mock_service_class.return_value
        mock_service.get_cache_stats.return_value = mock_stats

        response = client.get("/api/v1/cache/stats")

        assert response.status_code == 200
        data = response.json()
        assert data["cache_size"] == 5
        assert len(data["cached_simulations"]) == 2

    @patch("app.api.v1.endpoints.SimulationService")
    def test_clear_cache(self, mock_service_class):
        """Test clearing the cache."""
        mock_service = mock_service_class.return_value
        mock_service.clear_cache.return_value = None

        response = client.delete("/api/v1/cache/clear")

        assert response.status_code == 200
        assert response.json()["message"] == "Cache cleared successfully"

    @patch("app.api.v1.endpoints.SimulationService")
    def test_remove_from_cache(self, mock_service_class):
        """Test removing a specific simulation from cache."""
        mock_service = mock_service_class.return_value
        mock_service.remove_from_cache.return_value = True

        response = client.delete("/api/v1/cache/sim1")

        assert response.status_code == 200
        assert "Simulation sim1 removed from cache" in response.json()["message"]

    @patch("app.api.v1.endpoints.SimulationService")
    def test_remove_from_cache_not_found(self, mock_service_class):
        """Test removing a non-existent simulation from cache."""
        mock_service = mock_service_class.return_value
        mock_service.remove_from_cache.return_value = False

        response = client.delete("/api/v1/cache/nonexistent")

        assert response.status_code == 404
        assert "Simulation not found in cache" in response.json()["detail"]


class TestHealthEndpoint:
    """Test cases for health check endpoint."""

    def test_health_check(self):
        """Test health check endpoint returns correct response."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "f1-what-if-simulator"


class TestStartingGridEndpoint:
    """Test cases for starting grid endpoint."""

    @patch("app.api.v1.endpoints.SimulationService")
    def test_get_starting_grid_success(self, mock_service_class):
        """Test successful starting grid retrieval."""
        mock_service = AsyncMock()
        mock_service_class.return_value = mock_service

        # Mock grid data
        mock_grid_positions = [
            GridPositionResponse(
                position=1,
                driver_id=1,
                driver_name="Max Verstappen",
                driver_code="VER",
                team_name="Red Bull Racing",
                qualifying_time=78.241,
                qualifying_gap=0.0,
                qualifying_laps=3,
            ),
            GridPositionResponse(
                position=2,
                driver_id=2,
                driver_name="Lewis Hamilton",
                driver_code="HAM",
                team_name="Mercedes",
                qualifying_time=78.456,
                qualifying_gap=0.215,
                qualifying_laps=3,
            ),
        ]

        mock_grid = StartingGridResponse(
            session_key=12345,
            session_name="2024 Bahrain Grand Prix",
            track_name="Bahrain International Circuit",
            country="Bahrain",
            year=2024,
            total_drivers=2,
            grid_positions=mock_grid_positions,
        )

        mock_service.get_starting_grid.return_value = mock_grid

        response = client.get("/api/v1/grid/12345")
        assert response.status_code == 200
        data = response.json()

        assert data["session_key"] == 12345
        assert data["session_name"] == "2024 Bahrain Grand Prix"
        assert data["track_name"] == "Bahrain International Circuit"
        assert data["country"] == "Bahrain"
        assert data["year"] == 2024
        assert data["total_drivers"] == 2
        assert len(data["grid_positions"]) == 2

        # Verify first position
        first_pos = data["grid_positions"][0]
        assert first_pos["position"] == 1
        assert first_pos["driver_name"] == "Max Verstappen"
        assert first_pos["driver_code"] == "VER"
        assert first_pos["team_name"] == "Red Bull Racing"
        assert first_pos["qualifying_time"] == 78.241

    @patch("app.api.v1.endpoints.SimulationService")
    def test_get_starting_grid_invalid_session_key(self, mock_service_class):
        """Test starting grid with invalid session key."""
        mock_service = AsyncMock()
        mock_service_class.return_value = mock_service

        response = client.get("/api/v1/grid/invalid")
        assert response.status_code == 422  # Validation error

    @patch("app.api.v1.endpoints.SimulationService")
    def test_get_starting_grid_service_error(self, mock_service_class):
        """Test starting grid when service raises an error."""
        mock_service = AsyncMock()
        mock_service_class.return_value = mock_service
        mock_service.get_starting_grid.side_effect = Exception("Service error")

        response = client.get("/api/v1/grid/12345")
        assert response.status_code == 500
        data = response.json()
        assert "Failed to fetch starting grid" in data["detail"]


class TestGridSummaryEndpoint:
    """Test cases for grid summary endpoint."""

    @patch("app.api.v1.endpoints.SimulationService")
    def test_get_grid_summary_success(self, mock_service_class):
        """Test successful grid summary retrieval."""
        mock_service = AsyncMock()
        mock_service_class.return_value = mock_service

        # Mock pole position
        mock_pole_position = GridPositionResponse(
            position=1,
            driver_id=1,
            driver_name="Max Verstappen",
            driver_code="VER",
            team_name="Red Bull Racing",
            qualifying_time=78.241,
            qualifying_gap=0.0,
            qualifying_laps=3,
        )

        mock_summary = GridSummaryResponse(
            session_key=12345,
            pole_position=mock_pole_position,
            fastest_qualifying_time=78.241,
            slowest_qualifying_time=82.156,
            average_qualifying_time=80.198,
            time_gap_pole_to_last=3.915,
            teams_represented=["Red Bull Racing", "Mercedes", "Ferrari"],
        )

        mock_service.get_grid_summary.return_value = mock_summary

        response = client.get("/api/v1/grid/12345/summary")
        assert response.status_code == 200
        data = response.json()

        assert data["session_key"] == 12345
        assert data["fastest_qualifying_time"] == 78.241
        assert data["slowest_qualifying_time"] == 82.156
        assert data["average_qualifying_time"] == 80.198
        assert data["time_gap_pole_to_last"] == 3.915
        assert data["teams_represented"] == ["Red Bull Racing", "Mercedes", "Ferrari"]

        # Verify pole position
        pole_pos = data["pole_position"]
        assert pole_pos["position"] == 1
        assert pole_pos["driver_name"] == "Max Verstappen"
        assert pole_pos["driver_code"] == "VER"
        assert pole_pos["team_name"] == "Red Bull Racing"

    @patch("app.api.v1.endpoints.SimulationService")
    def test_get_grid_summary_invalid_session_key(self, mock_service_class):
        """Test grid summary with invalid session key."""
        mock_service = AsyncMock()
        mock_service_class.return_value = mock_service

        response = client.get("/api/v1/grid/invalid/summary")
        assert response.status_code == 422  # Validation error

    @patch("app.api.v1.endpoints.SimulationService")
    def test_get_grid_summary_service_error(self, mock_service_class):
        """Test grid summary when service raises an error."""
        mock_service = AsyncMock()
        mock_service_class.return_value = mock_service
        mock_service.get_grid_summary.side_effect = Exception("Service error")

        response = client.get("/api/v1/grid/12345/summary")
        assert response.status_code == 500
        data = response.json()
        assert "Failed to fetch grid summary" in data["detail"]


class TestLapTimesEndpoint:
    """Test cases for /api/v1/lap-times/{session_key} endpoint."""

    @patch("app.api.v1.endpoints.SimulationService")
    def test_get_lap_times_success(self, mock_service_class):
        """Test successful retrieval of lap times."""
        mock_lap_times = {
            "session_key": 12345,
            "session_name": "Monaco Grand Prix",
            "track_name": "Monaco",
            "country": "Monaco",
            "year": 2024,
            "total_laps": 50,
            "lap_times": [
                {
                    "lap_number": 1,
                    "driver_id": 1,
                    "driver_name": "Max Verstappen",
                    "driver_code": "VER",
                    "team_name": "Red Bull Racing",
                    "lap_time": 85.123,
                    "sector_1_time": 28.5,
                    "sector_2_time": 28.8,
                    "sector_3_time": 27.823,
                    "tire_compound": "soft",
                    "fuel_load": 100.0,
                    "lap_status": "valid",
                    "timestamp": "2024-05-26T14:01:00Z",
                }
            ],
        }
        mock_service = mock_service_class.return_value
        mock_service.get_lap_times = AsyncMock(return_value=mock_lap_times)

        response = client.get("/api/v1/lap-times/12345")

        assert response.status_code == 200
        data = response.json()
        assert data["session_key"] == 12345
        assert len(data["lap_times"]) == 1
        assert data["lap_times"][0]["lap_number"] == 1

    @patch("app.api.v1.endpoints.SimulationService")
    def test_get_lap_times_service_error(self, mock_service_class):
        """Test lap times endpoint when service raises an error."""
        mock_service = mock_service_class.return_value
        mock_service.get_lap_times = AsyncMock(side_effect=Exception("Service error"))

        response = client.get("/api/v1/lap-times/12345")

        assert response.status_code == 500
        assert "Failed to fetch lap times" in response.json()["detail"]


class TestPitStopsEndpoint:
    """Test cases for /api/v1/pit-stops/{session_key} endpoint."""

    @patch("app.api.v1.endpoints.SimulationService")
    def test_get_pit_stops_success(self, mock_service_class):
        """Test successful retrieval of pit stops."""
        mock_pit_stops = {
            "session_key": 12345,
            "session_name": "Monaco Grand Prix",
            "track_name": "Monaco",
            "country": "Monaco",
            "year": 2024,
            "total_pit_stops": 1,
            "pit_stops": [
                {
                    "pit_stop_number": 1,
                    "driver_id": 1,
                    "driver_name": "Max Verstappen",
                    "driver_code": "VER",
                    "team_name": "Red Bull Racing",
                    "lap_number": 15,
                    "pit_duration": 2.5,
                    "tire_compound_in": "medium",
                    "tire_compound_out": "soft",
                    "fuel_added": 20.0,
                    "pit_reason": "tire_change",
                    "timestamp": "2024-05-26T14:15:00Z",
                }
            ],
        }
        mock_service = mock_service_class.return_value
        mock_service.get_pit_stops = AsyncMock(return_value=mock_pit_stops)

        response = client.get("/api/v1/pit-stops/12345")

        assert response.status_code == 200
        data = response.json()
        assert data["session_key"] == 12345
        assert len(data["pit_stops"]) == 1
        assert data["pit_stops"][0]["lap_number"] == 15

    @patch("app.api.v1.endpoints.SimulationService")
    def test_get_pit_stops_service_error(self, mock_service_class):
        """Test pit stops endpoint when service raises an error."""
        mock_service = mock_service_class.return_value
        mock_service.get_pit_stops = AsyncMock(side_effect=Exception("Service error"))

        response = client.get("/api/v1/pit-stops/12345")

        assert response.status_code == 500
        assert "Failed to fetch pit stops" in response.json()["detail"]


class TestDriverPerformanceEndpoint:
    """Test cases for /api/v1/driver-performance/{session_key} endpoint."""

    @patch("app.api.v1.endpoints.SimulationService")
    def test_get_driver_performance_success(self, mock_service_class):
        """Test successful retrieval of driver performance."""
        mock_performance = {
            "session_key": 12345,
            "session_name": "Monaco Grand Prix",
            "track_name": "Monaco",
            "country": "Monaco",
            "year": 2024,
            "total_drivers": 20,
            "driver_performances": [
                {
                    "driver_id": 1,
                    "driver_name": "Max Verstappen",
                    "driver_code": "VER",
                    "team_name": "Red Bull Racing",
                    "total_laps": 50,
                    "best_lap_time": 85.123,
                    "avg_lap_time": 86.5,
                    "consistency_score": 0.95,
                    "total_pit_stops": 2,
                    "total_pit_time": 5.2,
                    "avg_pit_time": 2.6,
                    "tire_compounds_used": ["soft", "medium"],
                    "final_position": 1,
                    "race_status": "finished",
                }
            ],
        }
        mock_service = mock_service_class.return_value
        mock_service.get_driver_performance = AsyncMock(return_value=mock_performance)

        response = client.get("/api/v1/driver-performance/12345")

        assert response.status_code == 200
        data = response.json()
        assert data["session_key"] == 12345
        assert len(data["driver_performances"]) == 1
        assert data["driver_performances"][0]["driver_id"] == 1

    @patch("app.api.v1.endpoints.SimulationService")
    def test_get_driver_performance_service_error(self, mock_service_class):
        """Test driver performance endpoint when service raises an error."""
        mock_service = mock_service_class.return_value
        mock_service.get_driver_performance = AsyncMock(
            side_effect=Exception("Service error")
        )

        response = client.get("/api/v1/driver-performance/12345")

        assert response.status_code == 500
        assert "Failed to fetch driver performance" in response.json()["detail"]


class TestDataProcessingEndpoint:
    """Test cases for /api/v1/data-processing endpoint."""

    @patch("app.api.v1.endpoints.SimulationService")
    def test_process_session_data_success(self, mock_service_class):
        """Test successful data processing."""
        mock_response = {
            "session_key": 12345,
            "session_name": "Monaco Grand Prix",
            "track_name": "Monaco",
            "country": "Monaco",
            "year": 2024,
            "processing_summary": {
                "session_key": 12345,
                "total_data_points": 100,
                "total_drivers": 20,
                "total_laps": 50,
                "data_sources": ["weather", "grid", "lap_times"],
                "processing_time_ms": 150,
                "missing_data_points": 5,
                "data_quality_score": 0.95,
                "features_generated": ["lap_time_normalized"],
                "processing_errors": [],
            },
            "processed_data": [],
            "feature_columns": ["lap_number", "tire_compound"],
            "target_columns": ["lap_time"],
            "created_at": "2024-05-26T14:00:00Z",
        }
        mock_service = mock_service_class.return_value
        mock_service.process_session_data = AsyncMock(return_value=mock_response)

        request_data = {
            "session_key": 12345,
            "include_weather": True,
            "include_grid": True,
            "include_lap_times": True,
            "include_pit_stops": False,
            "processing_options": {},
        }

        response = client.post("/api/v1/data-processing", json=request_data)

        assert response.status_code == 200
        data = response.json()
        assert data["session_key"] == 12345
        assert data["processing_summary"]["total_data_points"] == 100

    @patch("app.api.v1.endpoints.SimulationService")
    def test_process_session_data_service_error(self, mock_service_class):
        """Test data processing when service raises an error."""
        mock_service = mock_service_class.return_value
        mock_service.process_session_data = AsyncMock(
            side_effect=Exception("Processing error")
        )

        request_data = {
            "session_key": 12345,
            "include_weather": True,
            "include_grid": True,
            "include_lap_times": True,
            "include_pit_stops": False,
            "processing_options": {},
        }

        response = client.post("/api/v1/data-processing", json=request_data)

        assert response.status_code == 500
        assert "Failed to process session data" in response.json()["detail"]


class TestFeatureEngineeringEndpoints:
    """Test cases for feature engineering endpoints."""

    @patch("app.api.v1.endpoints.SimulationService")
    def test_process_features_success(self, mock_service_class):
        """Test successful feature processing."""
        mock_service = AsyncMock()
        mock_service_class.return_value = mock_service

        # Mock the feature engineering service
        mock_feature_service = Mock()
        mock_service.feature_engineering_service = mock_feature_service

        # Mock the process_session_data method
        mock_processed_response = Mock()
        mock_processed_response.processed_data = []
        mock_processed_response.processing_summary.dict.return_value = {
            "total_data_points": 100,
            "total_drivers": 20,
            "data_quality_score": 0.95,
        }
        mock_service.process_session_data.return_value = mock_processed_response

        # Mock the feature engineering methods
        mock_feature_service.fit_transform_features.return_value = (
            np.array([[1, 2, 3], [4, 5, 6]]),  # features
            np.array([85.1, 84.9]),  # targets
            {"feature_columns": ["col1", "col2", "col3"]},  # metadata
        )
        mock_feature_service.get_feature_importance.return_value = {
            "col1": 0.8,
            "col2": 0.6,
            "col3": 0.4,
        }
        mock_feature_service.get_data_quality_report.return_value = {
            "total_records": 100,
            "data_quality_score": 0.95,
        }

        request_data = {
            "session_key": 12345,
            "include_weather": True,
            "include_grid": True,
            "include_lap_times": True,
            "include_pit_stops": True,
        }

        response = client.post("/api/v1/feature-engineering/process", json=request_data)
        assert response.status_code == 200
        data = response.json()

        assert data["session_key"] == 12345
        assert data["features_shape"] == [2, 3]
        assert data["targets_shape"] == [2]
        assert "feature_metadata" in data
        assert "feature_importance" in data
        assert "data_quality_report" in data

    @patch("app.api.v1.endpoints.SimulationService")
    def test_process_features_service_error(self, mock_service_class):
        """Test feature processing when service raises an error."""
        mock_service = AsyncMock()
        mock_service_class.return_value = mock_service
        mock_service.process_session_data.side_effect = Exception("Service error")

        request_data = {
            "session_key": 12345,
            "include_weather": True,
            "include_grid": True,
            "include_lap_times": True,
            "include_pit_stops": True,
        }

        response = client.post("/api/v1/feature-engineering/process", json=request_data)
        assert response.status_code == 500
        data = response.json()
        assert "Failed to process features" in data["detail"]

    @patch("app.api.v1.endpoints.SimulationService")
    def test_get_data_quality_report_success(self, mock_service_class):
        """Test successful data quality report retrieval."""
        mock_service = AsyncMock()
        mock_service_class.return_value = mock_service

        # Mock the process_session_data method
        mock_processed_response = Mock()
        mock_processed_response.processed_data = []
        mock_processed_response.processing_summary.dict.return_value = {
            "total_data_points": 100,
            "total_drivers": 20,
            "data_quality_score": 0.95,
        }
        mock_service.process_session_data.return_value = mock_processed_response

        # Mock the feature engineering service
        mock_feature_service = Mock()
        mock_service.feature_engineering_service = mock_feature_service
        mock_feature_service.get_data_quality_report.return_value = {
            "total_records": 100,
            "total_features": 15,
            "data_quality_score": 0.95,
            "missing_data_summary": {},
        }

        response = client.get("/api/v1/feature-engineering/quality-report/12345")
        assert response.status_code == 200
        data = response.json()

        assert data["session_key"] == 12345
        assert "quality_report" in data
        assert "processing_summary" in data
        assert data["quality_report"]["total_records"] == 100
        assert data["quality_report"]["data_quality_score"] == 0.95

    @patch("app.api.v1.endpoints.SimulationService")
    def test_get_feature_importance_success(self, mock_service_class):
        """Test successful feature importance retrieval."""
        mock_service = AsyncMock()
        mock_service_class.return_value = mock_service

        # Mock the get_feature_importance method
        mock_service.get_feature_importance = AsyncMock(
            return_value={
                "col1": 0.8,
                "col2": 0.6,
                "col3": 0.4,
            }
        )

        response = client.get("/api/v1/feature-engineering/feature-importance/12345")
        assert response.status_code == 200
        data = response.json()
        assert data["session_key"] == 12345
        assert "feature_importance" in data
        assert data["feature_importance"]["col1"] == 0.8
        assert data["feature_importance"]["col2"] == 0.6
        assert data["feature_importance"]["col3"] == 0.4

    @patch("app.api.v1.endpoints.SimulationService")
    def test_get_encoding_info(self, mock_service_class):
        """Test getting encoding information for a session."""
        # Mock the simulation service
        mock_service = MagicMock()
        mock_service.get_encoding_info = AsyncMock(
            return_value={
                "onehot_columns": ["tire_compound", "weather_condition"],
                "label_columns": ["lap_status"],
                "total_encoded_features": 3,
                "onehot_encoders": {
                    "tire_compound": {
                        "categories": ["soft", "medium", "hard"],
                        "n_features": 3,
                    }
                },
                "label_encoders": {
                    "lap_status": {"classes": ["valid", "invalid"], "n_classes": 2}
                },
            }
        )
        mock_service_class.return_value = mock_service

        response = client.get("/api/v1/feature-engineering/encoding-info/12345")

        assert response.status_code == 200
        data = response.json()
        assert data["session_key"] == 12345
        assert "encoding_info" in data
        assert "onehot_columns" in data
        assert "label_columns" in data
        assert "total_encoded_features" in data

    @patch("app.api.v1.endpoints.SimulationService")
    def test_apply_one_hot_encoding(self, mock_service_class):
        """Test applying one-hot encoding to categorical features."""
        # Mock the simulation service
        mock_service = MagicMock()
        mock_service.apply_one_hot_encoding = AsyncMock(
            return_value={
                "onehot_features_created": 6,
                "original_categorical_features": ["tire_compound", "weather_condition"],
                "new_feature_names": [
                    "tire_compound_soft",
                    "tire_compound_medium",
                    "tire_compound_hard",
                    "weather_condition_dry",
                    "weather_condition_wet",
                    "weather_condition_intermediate",
                ],
                "processing_time_ms": 150,
                "features_shape": [100, 15],
                "targets_shape": [100, 1],
            }
        )
        mock_service_class.return_value = mock_service

        request_data = {
            "session_key": 12345,
            "include_weather": True,
            "include_grid": True,
            "include_lap_times": True,
            "include_pit_stops": True,
            "processing_options": {"encoding_method": "onehot"},
        }

        response = client.post(
            "/api/v1/feature-engineering/one-hot-encode", json=request_data
        )

        assert response.status_code == 200
        data = response.json()
        assert data["session_key"] == 12345
        assert "encoding_result" in data
        assert "onehot_features_created" in data
        assert "original_categorical_features" in data
        assert "new_feature_names" in data
        assert "processing_time_ms" in data

    @patch("app.api.v1.endpoints.SimulationService")
    def test_get_encoding_statistics(self, mock_service_class):
        """Test getting encoding statistics for a session."""
        # Mock the simulation service
        mock_service = MagicMock()
        mock_service.get_encoding_statistics = AsyncMock(
            return_value={
                "total_categorical_features": 3,
                "onehot_encoded_features": 2,
                "label_encoded_features": 1,
                "feature_cardinality": {
                    "tire_compound": 3,
                    "weather_condition": 2,
                    "lap_status": 2,
                },
                "total_features_after_encoding": 15,
                "encoding_methods": {
                    "onehot": ["tire_compound", "weather_condition"],
                    "label": ["lap_status"],
                },
            }
        )
        mock_service_class.return_value = mock_service

        response = client.get("/api/v1/feature-engineering/encoding-stats/12345")

        assert response.status_code == 200
        data = response.json()
        assert data["session_key"] == 12345
        assert "encoding_statistics" in data
        assert "total_categorical_features" in data
        assert "onehot_encoded_features" in data
        assert "label_encoded_features" in data
        assert "feature_cardinality" in data

    @patch("app.api.v1.endpoints.SimulationService")
    def test_get_encoding_info_error(self, mock_service_class):
        """Test error handling for encoding info endpoint."""
        # Mock the simulation service to raise an error
        mock_service = MagicMock()
        mock_service.get_encoding_info = AsyncMock(
            side_effect=FeatureEngineeringError("Test error")
        )
        mock_service_class.return_value = mock_service

        response = client.get("/api/v1/feature-engineering/encoding-info/12345")

        assert response.status_code == 400
        data = response.json()
        assert "detail" in data

    @patch("app.api.v1.endpoints.SimulationService")
    def test_apply_one_hot_encoding_error(self, mock_service_class):
        """Test error handling for one-hot encoding endpoint."""
        # Mock the simulation service to raise an error
        mock_service = MagicMock()
        mock_service.apply_one_hot_encoding = AsyncMock(
            side_effect=FeatureEngineeringError("Test error")
        )
        mock_service_class.return_value = mock_service

        request_data = {
            "session_key": 12345,
            "include_weather": True,
            "include_grid": True,
            "include_lap_times": True,
            "include_pit_stops": True,
        }

        response = client.post(
            "/api/v1/feature-engineering/one-hot-encode", json=request_data
        )

        assert response.status_code == 400
        data = response.json()
        assert "detail" in data

    @patch("app.api.v1.endpoints.SimulationService")
    def test_get_encoding_statistics_error(self, mock_service_class):
        """Test error handling for encoding statistics endpoint."""
        # Mock the simulation service to raise an error
        mock_service = MagicMock()
        mock_service.get_encoding_statistics = AsyncMock(
            side_effect=FeatureEngineeringError("Test error")
        )
        mock_service_class.return_value = mock_service

        response = client.get("/api/v1/feature-engineering/encoding-stats/12345")

        assert response.status_code == 400
        data = response.json()
        assert "detail" in data

    @patch("app.api.v1.endpoints.SimulationService")
    def test_encode_categorical_feature_success(self, mock_service_class):
        """Test successful categorical feature encoding."""
        from app.api.v1.schemas import CategoricalMappingResponse

        mock_response = CategoricalMappingResponse(
            session_key=12345,
            feature_name="tire_compound",
            encoding_type="onehot",
            categories=["soft", "medium", "hard"],
            feature_mappings={
                "soft": [1, 0, 0],
                "medium": [0, 1, 0],
                "hard": [0, 0, 1],
            },
            encoded_feature_names=[
                "tire_compound_soft",
                "tire_compound_medium",
                "tire_compound_hard",
            ],
            encoding_metadata={
                "feature_count": 3,
                "sparse_encoding": False,
                "handle_unknown": "ignore",
                "encoding_version": "1.0",
                "encoding_timestamp": "2024-05-26T14:00:00Z",
            },
            validation_passed=True,
            created_at="2024-05-26T14:00:00Z",
        )
        mock_service = mock_service_class.return_value
        mock_service.encode_categorical_feature = AsyncMock(return_value=mock_response)

        request_data = {
            "session_key": 12345,
            "feature_name": "tire_compound",
            "encoding_type": "onehot",
            "include_validation": True,
        }

        response = client.post(
            "/api/v1/feature-engineering/categorical/encode", json=request_data
        )

        assert response.status_code == 200
        data = response.json()
        assert data["feature_name"] == "tire_compound"
        assert data["encoding_type"] == "onehot"
        assert len(data["categories"]) == 3

    @patch("app.api.v1.endpoints.SimulationService")
    def test_encode_categorical_feature_service_error(self, mock_service_class):
        """Test categorical encoding when service raises an error."""
        mock_service = mock_service_class.return_value
        mock_service.encode_categorical_feature = AsyncMock(
            side_effect=Exception("Encoding error")
        )

        request_data = {
            "session_key": 12345,
            "feature_name": "tire_compound",
            "encoding_type": "onehot",
            "include_validation": True,
        }

        response = client.post(
            "/api/v1/feature-engineering/categorical/encode", json=request_data
        )

        assert response.status_code == 500
        assert "Failed to encode categorical feature" in response.json()["detail"]

    @patch("app.api.v1.endpoints.SimulationService")
    def test_validate_categorical_encodings_success(self, mock_service_class):
        """Test successful categorical encodings validation."""
        mock_response = {
            "session_key": 12345,
            "total_features_validated": 2,
            "validation_passed": True,
            "feature_validations": {
                "tire_compound": True,
                "weather_condition": True,
            },
            "encoding_consistency": {
                "tire_compound": "consistent",
                "weather_condition": "consistent",
            },
            "validation_errors": [],
            "validation_time_ms": 150,
        }
        mock_service = mock_service_class.return_value
        mock_service.validate_categorical_encodings = AsyncMock(
            return_value=mock_response
        )

        request_data = {
            "session_key": 12345,
            "include_weather": True,
            "include_grid": True,
            "include_lap_times": True,
            "include_pit_stops": False,
            "processing_options": {},
        }

        response = client.post(
            "/api/v1/feature-engineering/categorical/validate", json=request_data
        )

        assert response.status_code == 200
        data = response.json()
        assert data["validation_passed"] is True
        assert data["total_features_validated"] == 2

    @patch("app.api.v1.endpoints.SimulationService")
    def test_validate_categorical_encodings_service_error(self, mock_service_class):
        """Test categorical validation when service raises an error."""
        mock_service = mock_service_class.return_value
        mock_service.validate_categorical_encodings = AsyncMock(
            side_effect=Exception("Validation error")
        )

        request_data = {
            "session_key": 12345,
            "include_weather": True,
            "include_grid": True,
            "include_lap_times": True,
            "include_pit_stops": False,
            "processing_options": {},
        }

        response = client.post(
            "/api/v1/feature-engineering/categorical/validate", json=request_data
        )

        assert response.status_code == 500
        assert "Failed to validate categorical encodings" in response.json()["detail"]

    @patch("app.api.v1.endpoints.SimulationService")
    def test_get_weather_condition_mappings(self, mock_service_class):
        """Test getting weather condition mappings."""
        from app.api.v1.schemas import CategoricalMappingResponse

        mock_response = CategoricalMappingResponse(
            session_key=12345,
            feature_name="weather_condition",
            encoding_type="onehot",
            categories=["dry", "wet", "intermediate"],
            feature_mappings={
                "dry": [1, 0, 0],
                "wet": [0, 1, 0],
                "intermediate": [0, 0, 1],
            },
            encoded_feature_names=[
                "weather_condition_dry",
                "weather_condition_wet",
                "weather_condition_intermediate",
            ],
            encoding_metadata={
                "feature_count": 3,
                "sparse_encoding": False,
                "handle_unknown": "ignore",
                "encoding_version": "1.0",
                "encoding_timestamp": "2024-05-26T14:00:00Z",
            },
            validation_passed=True,
            created_at="2024-05-26T14:00:00Z",
        )
        mock_service = mock_service_class.return_value
        mock_service.encode_categorical_feature = AsyncMock(return_value=mock_response)

        response = client.get(
            "/api/v1/feature-engineering/categorical/weather-mappings/12345"
        )

        assert response.status_code == 200
        data = response.json()
        assert data["feature_name"] == "weather_condition"
        assert len(data["categories"]) == 3

    @patch("app.api.v1.endpoints.SimulationService")
    def test_get_tire_compound_mappings(self, mock_service_class):
        """Test getting tire compound mappings."""
        from app.api.v1.schemas import CategoricalMappingResponse

        mock_response = CategoricalMappingResponse(
            session_key=12345,
            feature_name="tire_compound",
            encoding_type="onehot",
            categories=["soft", "medium", "hard"],
            feature_mappings={
                "soft": [1, 0, 0],
                "medium": [0, 1, 0],
                "hard": [0, 0, 1],
            },
            encoded_feature_names=[
                "tire_compound_soft",
                "tire_compound_medium",
                "tire_compound_hard",
            ],
            encoding_metadata={
                "feature_count": 3,
                "sparse_encoding": False,
                "handle_unknown": "ignore",
                "encoding_version": "1.0",
                "encoding_timestamp": "2024-05-26T14:00:00Z",
            },
            validation_passed=True,
            created_at="2024-05-26T14:00:00Z",
        )
        mock_service = mock_service_class.return_value
        mock_service.encode_categorical_feature = AsyncMock(return_value=mock_response)

        response = client.get(
            "/api/v1/feature-engineering/categorical/tire-mappings/12345"
        )

        assert response.status_code == 200
        data = response.json()
        assert data["feature_name"] == "tire_compound"
        assert len(data["categories"]) == 3

    @patch("app.api.v1.endpoints.SimulationService")
    def test_get_track_type_mappings(self, mock_service_class):
        """Test getting track type mappings."""
        from app.api.v1.schemas import CategoricalMappingResponse

        mock_response = CategoricalMappingResponse(
            session_key=12345,
            feature_name="track_type",
            encoding_type="onehot",
            categories=["street", "permanent", "hybrid"],
            feature_mappings={
                "street": [1, 0, 0],
                "permanent": [0, 1, 0],
                "hybrid": [0, 0, 1],
            },
            encoded_feature_names=[
                "track_type_street",
                "track_type_permanent",
                "track_type_hybrid",
            ],
            encoding_metadata={
                "feature_count": 3,
                "sparse_encoding": False,
                "handle_unknown": "ignore",
                "encoding_version": "1.0",
                "encoding_timestamp": "2024-05-26T14:00:00Z",
            },
            validation_passed=True,
            created_at="2024-05-26T14:00:00Z",
        )
        mock_service = mock_service_class.return_value
        mock_service.encode_categorical_feature = AsyncMock(return_value=mock_response)

        response = client.get(
            "/api/v1/feature-engineering/categorical/track-type-mappings/12345"
        )

        assert response.status_code == 200
        data = response.json()
        assert data["feature_name"] == "track_type"
        assert len(data["categories"]) == 3

    @patch("app.api.v1.endpoints.SimulationService")
    def test_get_driver_team_mappings(self, mock_service_class):
        """Test getting driver team mappings."""
        from app.api.v1.schemas import CategoricalMappingResponse

        mock_response = CategoricalMappingResponse(
            session_key=12345,
            feature_name="driver_team",
            encoding_type="onehot",
            categories=["Red Bull", "Mercedes", "Ferrari"],
            feature_mappings={
                "Red Bull": [1, 0, 0],
                "Mercedes": [0, 1, 0],
                "Ferrari": [0, 0, 1],
            },
            encoded_feature_names=[
                "driver_team_Red Bull",
                "driver_team_Mercedes",
                "driver_team_Ferrari",
            ],
            encoding_metadata={
                "feature_count": 3,
                "sparse_encoding": False,
                "handle_unknown": "ignore",
                "encoding_version": "1.0",
                "encoding_timestamp": "2024-05-26T14:00:00Z",
            },
            validation_passed=True,
            created_at="2024-05-26T14:00:00Z",
        )
        mock_service = mock_service_class.return_value
        mock_service.encode_categorical_feature = AsyncMock(return_value=mock_response)

        response = client.get(
            "/api/v1/feature-engineering/categorical/driver-team-mappings/12345"
        )

        assert response.status_code == 200
        data = response.json()
        assert data["feature_name"] == "driver_team"
        assert len(data["categories"]) == 3

    def test_simulation_request_season_validation(self):
        """Test season validation in SimulationRequest."""
        from pydantic import ValidationError

        # Test valid season
        valid_request = SimulationRequest(
            driver_id=1,
            track_id=1,
            season=2024,
            weather_conditions="dry",
            car_setup={},
        )
        assert valid_request.season == 2024

        # Test season too low
        with pytest.raises(ValidationError):
            SimulationRequest(
                driver_id=1,
                track_id=1,
                season=1949,
                weather_conditions="dry",
                car_setup={},
            )

        # Test season too high
        with pytest.raises(ValidationError):
            SimulationRequest(
                driver_id=1,
                track_id=1,
                season=2031,
                weather_conditions="dry",
                car_setup={},
            )

        # Test boundary values
        boundary_request_low = SimulationRequest(
            driver_id=1,
            track_id=1,
            season=1950,
            weather_conditions="dry",
            car_setup={},
        )
        assert boundary_request_low.season == 1950

        boundary_request_high = SimulationRequest(
            driver_id=1,
            track_id=1,
            season=2030,
            weather_conditions="dry",
            car_setup={},
        )
        assert boundary_request_high.season == 2030
