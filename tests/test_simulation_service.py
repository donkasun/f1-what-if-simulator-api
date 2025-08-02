"""
Tests for the simulation service.
"""

import pytest
from app.services.simulation_service import SimulationService
from app.api.v1.schemas import SimulationRequest


@pytest.mark.asyncio
async def test_simulation_service_initialization():
    """Test that the simulation service can be initialized."""
    service = SimulationService()
    assert service is not None
    assert hasattr(service, "_simulation_cache")


@pytest.mark.asyncio
async def test_get_drivers():
    """Test that drivers can be retrieved."""
    service = SimulationService()
    drivers = await service.get_drivers(2024)
    assert isinstance(drivers, list)
    assert len(drivers) > 0
    assert all(hasattr(driver, "driver_id") for driver in drivers)


@pytest.mark.asyncio
async def test_get_tracks():
    """Test that tracks can be retrieved."""
    service = SimulationService()
    tracks = await service.get_tracks(2024)
    assert isinstance(tracks, list)
    assert len(tracks) > 0
    assert all(hasattr(track, "track_id") for track in tracks)


@pytest.mark.asyncio
async def test_run_simulation():
    """Test that a simulation can be run."""
    service = SimulationService()
    request = SimulationRequest(
        season=2024,
        driver_id=1,
        track_id=1,
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
    service1 = SimulationService()
    service2 = SimulationService()

    request = SimulationRequest(
        season=2024,
        driver_id=1,
        track_id=1,
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

    # Test clear cache
    service.clear_cache()
    stats_after_clear = service.get_cache_stats()
    assert stats_after_clear["cache_size"] == 0
