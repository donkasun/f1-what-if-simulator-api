"""
Tests for API endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch
from app.main import app
from app.core.exceptions import InvalidSimulationParametersError

client = TestClient(app)


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
        mock_result = {
            "simulation_id": "sim_123",
            "predicted_lap_time": 85.234,
            "confidence_score": 0.92,
            "weather_conditions": "dry",
            "car_setup": {},
        }

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
    async def test_get_simulation_result_success(self):
        """Test successful retrieval of simulation result."""
        mock_result = {
            "simulation_id": "sim_123",
            "predicted_lap_time": 85.234,
            "confidence_score": 0.92,
            "weather_conditions": "dry",
            "car_setup": {},
        }

        with patch("app.api.v1.endpoints.SimulationService") as mock_service_class:
            mock_service = mock_service_class.return_value
            mock_service.get_simulation_result = AsyncMock(return_value=mock_result)

            response = client.get("/api/v1/simulate/sim_123")

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

            response = client.get("/api/v1/simulate/nonexistent")

            assert response.status_code == 400


class TestHealthEndpoint:
    """Test cases for health check endpoint."""

    def test_health_check(self):
        """Test health check endpoint."""
        response = client.get("/api/v1/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "f1-what-if-simulator"
