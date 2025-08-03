"""
Tests for the simulation service.
"""

import pytest
from unittest.mock import AsyncMock, patch
from app.services.simulation_service import SimulationService
from app.api.v1.schemas import SimulationRequest


@pytest.mark.asyncio
async def test_simulation_service_initialization():
    """Test that the simulation service can be initialized."""
    service = SimulationService()
    assert service is not None
    assert hasattr(service, "openf1_client")
    assert hasattr(service, "model_loader")


@pytest.mark.asyncio
async def test_get_drivers():
    """Test that drivers can be retrieved."""
    with patch("app.services.simulation_service.OpenF1Client") as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client

        # Mock the context manager
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None

        # Mock the get_drivers method
        mock_client.get_drivers.return_value = [
            {
                "driver_number": 1,
                "full_name": "Max VERSTAPPEN",
                "name_acronym": "VER",
                "team_name": "Red Bull Racing",
                "country_code": "NED",
            },
            {
                "driver_number": 2,
                "full_name": "Logan SARGEANT",
                "name_acronym": "SAR",
                "team_name": "Williams",
                "country_code": "USA",
            },
        ]

        service = SimulationService()
        drivers = await service.get_drivers(2024)
        assert isinstance(drivers, list)
        assert len(drivers) > 0
        assert all(hasattr(driver, "driver_id") for driver in drivers)


@pytest.mark.asyncio
async def test_get_tracks():
    """Test that tracks can be retrieved."""
    with patch("app.services.simulation_service.OpenF1Client") as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client

        # Mock the context manager
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None

        # Mock the get_tracks method
        mock_client.get_tracks.return_value = [
            {
                "track_id": 10,
                "name": "Melbourne",
                "country": "Australia",
                "circuit_length": 5.303,
                "number_of_laps": 58,
            },
            {
                "track_id": 63,
                "name": "Sakhir",
                "country": "Bahrain",
                "circuit_length": 5.412,
                "number_of_laps": 57,
            },
        ]

        service = SimulationService()
        tracks = await service.get_tracks(2024)
        assert isinstance(tracks, list)
        assert len(tracks) > 0
        assert all(hasattr(track, "track_id") for track in tracks)


@pytest.mark.asyncio
async def test_run_simulation():
    """Test that a simulation can be run."""
    with patch("app.services.simulation_service.OpenF1Client") as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client

        # Mock the context manager
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None

        # Mock the get_drivers method
        mock_client.get_drivers.return_value = [
            {
                "driver_number": 1,
                "full_name": "Max VERSTAPPEN",
                "name_acronym": "VER",
                "team_name": "Red Bull Racing",
                "country_code": "NED",
            },
            {
                "driver_number": 2,
                "full_name": "Logan SARGEANT",
                "name_acronym": "SAR",
                "team_name": "Williams",
                "country_code": "USA",
            },
        ]

        # Mock the get_tracks method
        mock_client.get_tracks.return_value = [
            {
                "track_id": 10,
                "name": "Melbourne",
                "country": "Australia",
                "circuit_length": 5.303,
                "number_of_laps": 58,
            },
            {
                "track_id": 63,
                "name": "Sakhir",
                "country": "Bahrain",
                "circuit_length": 5.412,
                "number_of_laps": 57,
            },
        ]

        # Mock the get_historical_data method
        mock_client.get_historical_data.return_value = {
            "total_laps": 50,
            "avg_lap_time": 85.5,
            "best_lap_time": 82.3,
            "avg_i2_speed": 240.0,
            "avg_speed_trap": 320.0,
            "consistency_score": 0.85,
        }

        service = SimulationService()
        request = SimulationRequest(
            season=2024,
            driver_id=1,
            track_id=10,  # Melbourne track (valid track ID)
            weather_conditions="dry",
            starting_position=1,
            car_setup={},
        )

        result = await service.run_simulation(request)
        assert result is not None
        assert hasattr(result, "simulation_id")
        assert hasattr(result, "predicted_lap_time")
        assert hasattr(result, "confidence_score")


@pytest.mark.asyncio
async def test_simulation_caching():
    """Test that simulation results are cached."""
    with patch("app.services.simulation_service.OpenF1Client") as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value = mock_client

        # Mock the context manager
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None

        # Mock the get_drivers method
        mock_client.get_drivers.return_value = [
            {
                "driver_number": 1,
                "full_name": "Max VERSTAPPEN",
                "name_acronym": "VER",
                "team_name": "Red Bull Racing",
                "country_code": "NED",
            },
            {
                "driver_number": 2,
                "full_name": "Logan SARGEANT",
                "name_acronym": "SAR",
                "team_name": "Williams",
                "country_code": "USA",
            },
        ]

        # Mock the get_tracks method
        mock_client.get_tracks.return_value = [
            {
                "track_id": 10,
                "name": "Melbourne",
                "country": "Australia",
                "circuit_length": 5.303,
                "number_of_laps": 58,
            },
            {
                "track_id": 63,
                "name": "Sakhir",
                "country": "Bahrain",
                "circuit_length": 5.412,
                "number_of_laps": 57,
            },
        ]

        # Mock the get_historical_data method
        mock_client.get_historical_data.return_value = {
            "total_laps": 50,
            "avg_lap_time": 85.5,
            "best_lap_time": 82.3,
            "avg_i2_speed": 240.0,
            "avg_speed_trap": 320.0,
            "consistency_score": 0.85,
        }

        service1 = SimulationService()
        service2 = SimulationService()

        request = SimulationRequest(
            season=2024,
            driver_id=1,
            track_id=10,  # Melbourne track (valid track ID)
            weather_conditions="dry",
            starting_position=1,
            car_setup={},
        )

        # Run simulation with service1
        result1 = await service1.run_simulation(request)
        simulation_id = result1.simulation_id

        # Try to retrieve with service2 (should work with global cache)
        result2 = await service2.get_simulation_result(simulation_id)
        assert result2.simulation_id == simulation_id


@pytest.mark.asyncio
async def test_cache_management():
    """Test cache management functions."""
    service = SimulationService()

    # Test cache stats
    stats = service.get_cache_stats()
    assert isinstance(stats, dict)
    assert "cache_size" in stats
    assert "cached_simulations" in stats

    # Test cache clearing
    service.clear_cache()
    stats_after_clear = service.get_cache_stats()
    assert stats_after_clear["cache_size"] == 0

    # Test removing non-existent simulation
    result = service.remove_from_cache("non_existent")
    assert result is False
