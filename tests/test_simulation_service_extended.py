"""
Extended tests for the simulation service.
"""

import pytest
from unittest.mock import AsyncMock, patch
from app.services.simulation_service import SimulationService
from app.api.v1.schemas import (
    SimulationRequest,
    SessionResponse,
    WeatherDataResponse,
    WeatherSummaryResponse,
)
from app.external.openf1_client import OpenF1APIError
from app.core.exceptions import DriverNotFoundError, InvalidSimulationParametersError


class TestSimulationServiceWeatherData:
    """Test cases for weather data methods in simulation service."""

    @pytest.mark.asyncio
    async def test_get_sessions_success(self):
        """Test successful session retrieval through service."""
        mock_sessions_data = [
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

        with patch("app.services.simulation_service.OpenF1Client") as mock_client_class:
            mock_client = mock_client_class.return_value
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.get_sessions = AsyncMock(return_value=mock_sessions_data)

            service = SimulationService()
            result = await service.get_sessions(2024)

            assert isinstance(result, list)
            assert len(result) == 1
            assert isinstance(result[0], SessionResponse)
            assert result[0].session_key == 12345
            assert result[0].location == "Monaco"

    @pytest.mark.asyncio
    async def test_get_sessions_api_error(self):
        """Test session retrieval with API error."""
        with patch("app.services.simulation_service.OpenF1Client") as mock_client_class:
            mock_client = mock_client_class.return_value
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.get_sessions = AsyncMock(
                side_effect=OpenF1APIError("API error")
            )

            service = SimulationService()
            with pytest.raises(OpenF1APIError, match="API error"):
                await service.get_sessions(2024)

    @pytest.mark.asyncio
    async def test_get_weather_data_success(self):
        """Test successful weather data retrieval through service."""
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

        with patch("app.services.simulation_service.OpenF1Client") as mock_client_class:
            mock_client = mock_client_class.return_value
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.get_weather_data = AsyncMock(return_value=mock_weather_data)

            service = SimulationService()
            result = await service.get_weather_data(12345)

            assert isinstance(result, list)
            assert len(result) == 1
            assert isinstance(result[0], WeatherDataResponse)
            assert result[0].session_key == 12345
            assert result[0].air_temperature == 25.5

    @pytest.mark.asyncio
    async def test_get_weather_data_api_error(self):
        """Test weather data retrieval with API error."""
        with patch("app.services.simulation_service.OpenF1Client") as mock_client_class:
            mock_client = mock_client_class.return_value
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.get_weather_data = AsyncMock(
                side_effect=OpenF1APIError("API error")
            )

            service = SimulationService()
            with pytest.raises(OpenF1APIError, match="API error"):
                await service.get_weather_data(12345)

    @pytest.mark.asyncio
    async def test_get_session_weather_summary_success(self):
        """Test successful weather summary retrieval through service."""
        mock_summary_data = {
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

        with patch("app.services.simulation_service.OpenF1Client") as mock_client_class:
            mock_client = mock_client_class.return_value
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.get_session_weather_summary = AsyncMock(
                return_value=mock_summary_data
            )

            service = SimulationService()
            result = await service.get_session_weather_summary(12345)

            assert isinstance(result, WeatherSummaryResponse)
            assert result.session_key == 12345
            assert result.weather_condition == "dry"
            assert result.avg_air_temperature == 25.5
            assert result.data_points == 10

    @pytest.mark.asyncio
    async def test_get_session_weather_summary_api_error(self):
        """Test weather summary retrieval with API error."""
        with patch("app.services.simulation_service.OpenF1Client") as mock_client_class:
            mock_client = mock_client_class.return_value
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.get_session_weather_summary = AsyncMock(
                side_effect=OpenF1APIError("API error")
            )

            service = SimulationService()
            with pytest.raises(OpenF1APIError, match="API error"):
                await service.get_session_weather_summary(12345)


class TestSimulationServiceExtended:
    """Extended test cases for simulation service."""

    @pytest.mark.asyncio
    async def test_simulation_with_real_weather_data(self):
        """Test simulation with real weather data integration."""
        mock_weather_summary = {
            "session_key": 12345,
            "weather_condition": "wet",
            "avg_air_temperature": 20.0,
            "avg_track_temperature": 25.0,
            "avg_humidity": 85.0,
            "avg_pressure": 1000.0,
            "avg_wind_speed": 10.0,
            "total_rainfall": 5.0,
            "data_points": 5,
        }

        with patch("app.services.simulation_service.OpenF1Client") as mock_client_class:
            mock_client = mock_client_class.return_value
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=None)
            mock_client.get_session_weather_summary = AsyncMock(
                return_value=mock_weather_summary
            )

            service = SimulationService()
            request = SimulationRequest(
                season=2024,
                driver_id=1,
                track_id=1,
                weather_conditions="wet",
                starting_position=1,
                car_setup={},
            )

            result = await service.run_simulation(request)

            assert result is not None
            assert hasattr(result, "simulation_id")
            assert hasattr(result, "predicted_lap_time")
            assert hasattr(result, "confidence_score")

    @pytest.mark.asyncio
    async def test_simulation_cache_with_weather_data(self):
        """Test that simulations with weather data are properly cached."""
        service1 = SimulationService()
        service2 = SimulationService()

        request = SimulationRequest(
            season=2024,
            driver_id=1,
            track_id=1,
            weather_conditions="intermediate",
            starting_position=5,
            car_setup={"downforce": "high", "tire_pressure": "medium"},
        )

        # Run simulation with service1
        result1 = await service1.run_simulation(request)
        simulation_id = result1.simulation_id

        # Try to retrieve with service2 (should work with global cache)
        result2 = await service2.get_simulation_result(simulation_id)
        assert result2.simulation_id == simulation_id

    @pytest.mark.asyncio
    async def test_simulation_with_complex_car_setup(self):
        """Test simulation with complex car setup data."""
        service = SimulationService()
        request = SimulationRequest(
            season=2024,
            driver_id=2,
            track_id=3,
            weather_conditions="dry",
            starting_position=10,
            car_setup={
                "downforce": "low",
                "tire_pressure": "high",
                "suspension": "soft",
                "brake_bias": "rear",
                "differential": "open",
            },
        )

        result = await service.run_simulation(request)

        assert result is not None
        assert hasattr(result, "simulation_id")
        assert hasattr(result, "predicted_lap_time")
        assert hasattr(result, "confidence_score")
        assert hasattr(result, "car_setup")

    @pytest.mark.asyncio
    async def test_simulation_error_handling(self):
        """Test simulation error handling."""
        service = SimulationService()

        # Test with invalid driver_id
        request = SimulationRequest(
            season=2024,
            driver_id=999,  # Invalid driver ID
            track_id=1,
            weather_conditions="dry",
            starting_position=1,
            car_setup={},
        )

        # Should raise DriverNotFoundError
        with pytest.raises(DriverNotFoundError):
            await service.run_simulation(request)

    @pytest.mark.asyncio
    async def test_cache_stats_with_weather_data(self):
        """Test cache statistics with weather data simulations."""
        service = SimulationService()

        # Run multiple simulations with different weather conditions
        weather_conditions = ["dry", "wet", "intermediate"]

        for weather in weather_conditions:
            request = SimulationRequest(
                season=2024,
                driver_id=1,
                track_id=1,
                weather_conditions=weather,
                starting_position=1,
                car_setup={},
            )
            await service.run_simulation(request)

        # Check cache stats
        stats = service.get_cache_stats()
        assert isinstance(stats, dict)
        assert "cache_size" in stats
        assert "cached_simulations" in stats
        assert stats["cache_size"] >= 3

    @pytest.mark.asyncio
    async def test_simulation_with_different_starting_positions(self):
        """Test simulations with different starting positions."""
        service = SimulationService()

        positions = [1, 5, 10, 20]
        results = []

        for position in positions:
            request = SimulationRequest(
                season=2024,
                driver_id=1,
                track_id=1,
                weather_conditions="dry",
                starting_position=position,
                car_setup={},
            )
            result = await service.run_simulation(request)
            results.append(result)

        # All simulations should complete successfully
        assert len(results) == 4
        for result in results:
            assert result is not None
            assert hasattr(result, "simulation_id")

    @pytest.mark.asyncio
    async def test_simulation_result_retrieval_edge_cases(self):
        """Test simulation result retrieval edge cases."""
        service = SimulationService()

        # Test retrieving non-existent simulation
        with pytest.raises(InvalidSimulationParametersError):
            await service.get_simulation_result("nonexistent_id")

        # Test retrieving with empty string
        with pytest.raises(InvalidSimulationParametersError):
            await service.get_simulation_result("")

        # Test retrieving with None
        with pytest.raises(InvalidSimulationParametersError):
            await service.get_simulation_result(None)

    @pytest.mark.asyncio
    async def test_simulation_service_initialization_with_client(self):
        """Test simulation service initialization with OpenF1 client."""
        service = SimulationService()

        assert service is not None
        assert hasattr(service, "_simulation_cache")
        assert hasattr(service, "openf1_client")
        assert service.openf1_client is not None

    @pytest.mark.asyncio
    async def test_simulation_with_all_weather_conditions(self):
        """Test simulations with all weather condition types."""
        service = SimulationService()
        weather_conditions = ["dry", "wet", "intermediate"]

        for weather in weather_conditions:
            request = SimulationRequest(
                season=2024,
                driver_id=1,
                track_id=1,
                weather_conditions=weather,
                starting_position=1,
                car_setup={},
            )

            result = await service.run_simulation(request)
            assert result is not None
            assert result.weather_conditions == weather
