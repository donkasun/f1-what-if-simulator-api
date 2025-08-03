"""
Tests for the simulation service.
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from app.services.simulation_service import SimulationService
from app.api.v1.schemas import SimulationRequest
import numpy as np


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


@pytest.mark.asyncio
async def test_process_session_data_success():
    """Test successful session data processing."""
    from app.api.v1.schemas import DataProcessingRequest

    mock_session_info = {
        "session_key": 12345,
        "session_name": "Race",
        "track_name": "Monaco",
        "country": "Monaco",
        "year": 2024,
    }

    mock_weather_data = {
        "session_key": 12345,
        "weather_condition": "dry",
        "avg_air_temperature": 25.0,
        "avg_track_temperature": 35.0,
        "avg_humidity": 60.0,
        "avg_pressure": 1000.0,
        "avg_wind_speed": 10.0,
        "total_rainfall": 0.0,
        "data_points": 5,
    }

    mock_grid_data = {
        "session_key": 12345,
        "session_name": "Race",
        "track_name": "Monaco",
        "country": "Monaco",
        "year": 2024,
        "total_drivers": 20,
        "grid_positions": [
            {
                "position": 1,
                "driver_id": 1,
                "driver_name": "Max Verstappen",
                "driver_code": "VER",
                "team_name": "Red Bull Racing",
                "qualifying_time": 78.456,
                "qualifying_gap": 0.0,
                "qualifying_laps": 3,
            }
        ],
    }

    mock_lap_times_data = {
        "session_key": 12345,
        "session_name": "Race",
        "track_name": "Monaco",
        "country": "Monaco",
        "year": 2024,
        "total_laps": 78,
        "lap_times": [
            {
                "lap_number": 1,
                "driver_id": 1,
                "driver_name": "Max Verstappen",
                "driver_code": "VER",
                "team_name": "Red Bull Racing",
                "lap_time": 85.234,
                "sector_1_time": 28.5,
                "sector_2_time": 29.2,
                "sector_3_time": 27.5,
                "tire_compound": "soft",
                "fuel_load": 100.0,
                "lap_status": "valid",
                "timestamp": "2024-05-26T14:00:00Z",
            }
        ],
    }

    mock_pit_stops_data = {
        "session_key": 12345,
        "session_name": "Race",
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
                "lap_number": 25,
                "pit_duration": 2.5,
                "tire_compound_in": "medium",
                "tire_compound_out": "soft",
                "fuel_added": 50.0,
                "pit_reason": "tire_change",
                "timestamp": "2024-05-26T14:30:00Z",
            }
        ],
    }

    with (
        patch("app.services.simulation_service.OpenF1Client") as mock_client_class,
        patch(
            "app.services.simulation_service.FeatureEngineeringService"
        ) as mock_fe_service_class,
    ):
        mock_client = mock_client_class.return_value
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        # Mock client methods
        mock_client.get_sessions = AsyncMock(return_value=[mock_session_info])
        mock_client.get_session_weather_summary = AsyncMock(
            return_value=mock_weather_data
        )
        mock_client.get_session_grid_summary = AsyncMock(return_value=mock_grid_data)
        mock_client.get_session_lap_times_summary = AsyncMock(
            return_value=mock_lap_times_data
        )
        mock_client.get_session_pit_stops_summary = AsyncMock(
            return_value=mock_pit_stops_data
        )

        # Mock feature engineering service
        mock_fe_service = mock_fe_service_class.return_value
        mock_fe_service.fit_transform_features = MagicMock(
            return_value=(
                np.array([[1.0, 2.0, 3.0]]),  # features
                np.array([85.0]),  # targets
                {"feature_columns": ["col1", "col2", "col3"]},  # metadata
            )
        )
        mock_fe_service.get_data_quality_report = MagicMock(
            return_value={
                "data_quality_score": 0.95,
                "missing_data_summary": {
                    "lap_time": {"missing_count": 0, "missing_percentage": 0.0},
                    "air_temperature": {"missing_count": 0, "missing_percentage": 0.0},
                },
            }
        )

        service = SimulationService()

        request = DataProcessingRequest(
            session_key=12345,
            include_weather=True,
            include_grid=True,
            include_lap_times=True,
            include_pit_stops=True,
            processing_options={"feature_engineering": True},
        )

        result = await service.process_session_data(request)

        assert result.session_key == 12345
        assert result.session_name == "Race"
        assert result.track_name == "Monaco"
        assert result.country == "Monaco"
        assert result.year == 2024
        assert len(result.processed_data) > 0
        assert result.feature_columns == ["col1", "col2", "col3"]
        assert "lap_time" in result.target_columns
        assert "sector_1_time" in result.target_columns
        assert "sector_2_time" in result.target_columns
        assert "sector_3_time" in result.target_columns


@pytest.mark.asyncio
async def test_process_session_data_session_not_found():
    """Test session data processing when session is not found."""
    from app.api.v1.schemas import DataProcessingRequest

    with (patch("app.services.simulation_service.OpenF1Client") as mock_client_class,):
        mock_client = mock_client_class.return_value
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        # Mock client to return empty sessions
        mock_client.get_sessions = AsyncMock(return_value=[])

        service = SimulationService()

        request = DataProcessingRequest(
            session_key=99999,
            include_weather=True,
            include_grid=True,
            include_lap_times=True,
            include_pit_stops=True,
        )

        with pytest.raises(ValueError, match="Session 99999 not found"):
            await service.process_session_data(request)


@pytest.mark.asyncio
async def test_encode_categorical_feature_success():
    """Test successful categorical feature encoding."""
    from app.api.v1.schemas import (
        CategoricalEncodingRequest,
    )

    mock_encoding_result = {
        "feature_name": "tire_compound",
        "encoding_type": "onehot",
        "categories": ["soft", "medium", "hard"],
        "feature_mappings": {
            "soft": [1, 0, 0],
            "medium": [0, 1, 0],
            "hard": [0, 0, 1],
        },
        "encoded_feature_names": [
            "tire_compound_soft",
            "tire_compound_medium",
            "tire_compound_hard",
        ],
        "encoding_metadata": {"cardinality": 3},
        "validation_passed": True,
    }

    with (
        patch("app.services.simulation_service.OpenF1Client") as mock_client_class,
        patch(
            "app.services.simulation_service.FeatureEngineeringService"
        ) as mock_fe_service_class,
    ):
        mock_client = mock_client_class.return_value
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        # Mock client methods for process_session_data
        mock_client.get_sessions = AsyncMock(
            return_value=[
                {
                    "session_key": 12345,
                    "session_name": "Race",
                    "track_name": "Monaco",
                    "country": "Monaco",
                    "year": 2024,
                }
            ]
        )
        mock_client.get_session_weather_summary = AsyncMock(return_value={})
        mock_client.get_session_grid_summary = AsyncMock(return_value={})
        mock_client.get_session_lap_times_summary = AsyncMock(return_value={})
        mock_client.get_session_pit_stops_summary = AsyncMock(return_value={})

        # Mock feature engineering service
        mock_fe_service = mock_fe_service_class.return_value
        mock_fe_service.fit_transform_features = MagicMock(
            return_value=(
                np.array([[1.0, 2.0, 3.0]]),  # features
                np.array([85.0]),  # targets
                {"feature_columns": ["col1", "col2", "col3"]},  # metadata
            )
        )
        mock_fe_service.encode_categorical_feature = MagicMock(
            return_value=mock_encoding_result
        )

        service = SimulationService()

        request = CategoricalEncodingRequest(
            session_key=12345,
            feature_name="tire_compound",
            encoding_type="onehot",
            include_validation=True,
        )

        result = await service.encode_categorical_feature(request)

        assert result.session_key == 12345
        assert result.feature_name == "tire_compound"
        assert result.encoding_type == "onehot"
        assert result.categories == ["soft", "medium", "hard"]
        assert result.validation_passed is True


@pytest.mark.asyncio
async def test_validate_categorical_encodings_success():
    """Test successful categorical encoding validation."""
    from app.api.v1.schemas import DataProcessingRequest

    mock_validation_result = {
        "total_features_validated": 3,
        "validation_passed": True,
        "feature_validations": {
            "tire_compound": True,
            "weather_condition": True,
            "driver_team": True,
        },
        "encoding_consistency": {
            "tire_compound": "consistent",
            "weather_condition": "consistent",
            "driver_team": "consistent",
        },
        "validation_errors": [],
        "validation_time_ms": 150,
    }

    with (
        patch("app.services.simulation_service.OpenF1Client") as mock_client_class,
        patch(
            "app.services.simulation_service.FeatureEngineeringService"
        ) as mock_fe_service_class,
    ):
        mock_client = mock_client_class.return_value
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        # Mock client methods for process_session_data
        mock_client.get_sessions = AsyncMock(
            return_value=[
                {
                    "session_key": 12345,
                    "session_name": "Race",
                    "track_name": "Monaco",
                    "country": "Monaco",
                    "year": 2024,
                }
            ]
        )
        mock_client.get_session_weather_summary = AsyncMock(return_value={})
        mock_client.get_session_grid_summary = AsyncMock(return_value={})
        mock_client.get_session_lap_times_summary = AsyncMock(return_value={})
        mock_client.get_session_pit_stops_summary = AsyncMock(return_value={})

        # Mock feature engineering service
        mock_fe_service = mock_fe_service_class.return_value
        mock_fe_service.fit_transform_features = MagicMock(
            return_value=(
                np.array([[1.0, 2.0, 3.0]]),  # features
                np.array([85.0]),  # targets
                {"feature_columns": ["col1", "col2", "col3"]},  # metadata
            )
        )
        mock_fe_service.validate_categorical_encodings = MagicMock(
            return_value=mock_validation_result
        )

        service = SimulationService()

        request = DataProcessingRequest(
            session_key=12345,
            include_weather=True,
            include_grid=True,
            include_lap_times=True,
            include_pit_stops=True,
            processing_options={"validate_encodings": True},
        )

        result = await service.validate_categorical_encodings(request)

        assert result.session_key == 12345
        assert result.total_features_validated == 3
        assert result.validation_passed is True
        assert len(result.feature_validations) == 3
        assert len(result.validation_errors) == 0


@pytest.mark.asyncio
async def test_get_feature_importance_success():
    """Test successful feature importance retrieval."""
    with (
        patch("app.services.simulation_service.OpenF1Client") as mock_client_class,
        patch(
            "app.services.simulation_service.FeatureEngineeringService"
        ) as mock_fe_service_class,
    ):
        mock_client = mock_client_class.return_value
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        # Mock client methods for process_session_data
        mock_client.get_sessions = AsyncMock(
            return_value=[
                {
                    "session_key": 12345,
                    "session_name": "Race",
                    "track_name": "Monaco",
                    "country": "Monaco",
                    "year": 2024,
                }
            ]
        )
        mock_client.get_session_weather_summary = AsyncMock(return_value={})
        mock_client.get_session_grid_summary = AsyncMock(return_value={})
        mock_client.get_session_lap_times_summary = AsyncMock(return_value={})
        mock_client.get_session_pit_stops_summary = AsyncMock(return_value={})

        # Mock feature engineering service
        mock_fe_service = mock_fe_service_class.return_value
        mock_fe_service.fit_transform_features = MagicMock(
            return_value=(
                np.array([[1.0, 2.0, 3.0]]),  # features
                np.array([85.0]),  # targets
                {"feature_columns": ["col1", "col2", "col3"]},  # metadata
            )
        )
        mock_fe_service.get_feature_importance = MagicMock(
            return_value={
                "air_temperature": 0.85,
                "track_temperature": 0.72,
                "humidity": 0.45,
                "tire_compound": 0.91,
                "fuel_load": 0.38,
            }
        )

        service = SimulationService()

        result = await service.get_feature_importance(12345)

        assert result["air_temperature"] == 0.85
        assert result["track_temperature"] == 0.72
        assert result["humidity"] == 0.45
        assert result["tire_compound"] == 0.91
        assert result["fuel_load"] == 0.38


@pytest.mark.asyncio
async def test_get_encoding_info_success():
    """Test successful encoding info retrieval."""
    with (
        patch("app.services.simulation_service.OpenF1Client") as mock_client_class,
        patch(
            "app.services.simulation_service.FeatureEngineeringService"
        ) as mock_fe_service_class,
    ):
        mock_client = mock_client_class.return_value
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        # Mock client methods for process_session_data
        mock_client.get_sessions = AsyncMock(
            return_value=[
                {
                    "session_key": 12345,
                    "session_name": "Race",
                    "track_name": "Monaco",
                    "country": "Monaco",
                    "year": 2024,
                }
            ]
        )
        mock_client.get_session_weather_summary = AsyncMock(return_value={})
        mock_client.get_session_grid_summary = AsyncMock(return_value={})
        mock_client.get_session_lap_times_summary = AsyncMock(return_value={})
        mock_client.get_session_pit_stops_summary = AsyncMock(return_value={})

        # Mock feature engineering service
        mock_fe_service = mock_fe_service_class.return_value
        mock_fe_service.fit_transform_features = MagicMock(
            return_value=(
                np.array([[1.0, 2.0, 3.0]]),  # features
                np.array([85.0]),  # targets
                {"feature_columns": ["col1", "col2", "col3"]},  # metadata
            )
        )
        mock_fe_service.get_encoding_info = MagicMock(
            return_value={
                "categorical_features": [
                    "tire_compound",
                    "weather_condition",
                    "driver_team",
                ],
                "encoding_types": {
                    "tire_compound": "onehot",
                    "weather_condition": "onehot",
                    "driver_team": "label",
                },
                "feature_cardinalities": {
                    "tire_compound": 3,
                    "weather_condition": 2,
                    "driver_team": 10,
                },
                "encoded_feature_count": 15,
            }
        )

        service = SimulationService()

        result = await service.get_encoding_info(12345)

        assert result["categorical_features"] == [
            "tire_compound",
            "weather_condition",
            "driver_team",
        ]
        assert result["encoding_types"]["tire_compound"] == "onehot"
        assert result["feature_cardinalities"]["tire_compound"] == 3
        assert result["encoded_feature_count"] == 15


@pytest.mark.asyncio
async def test_apply_one_hot_encoding_success():
    """Test successful one-hot encoding application."""
    from app.api.v1.schemas import DataProcessingRequest

    with (
        patch("app.services.simulation_service.OpenF1Client") as mock_client_class,
        patch(
            "app.services.simulation_service.FeatureEngineeringService"
        ) as mock_fe_service_class,
    ):
        mock_client = mock_client_class.return_value
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        # Mock client methods for process_session_data
        mock_client.get_sessions = AsyncMock(
            return_value=[
                {
                    "session_key": 12345,
                    "session_name": "Race",
                    "track_name": "Monaco",
                    "country": "Monaco",
                    "year": 2024,
                }
            ]
        )
        mock_client.get_session_weather_summary = AsyncMock(return_value={})
        mock_client.get_session_grid_summary = AsyncMock(return_value={})
        mock_client.get_session_lap_times_summary = AsyncMock(return_value={})
        mock_client.get_session_pit_stops_summary = AsyncMock(return_value={})

        # Mock feature engineering service
        mock_fe_service = mock_fe_service_class.return_value
        mock_fe_service.fit_transform_features = MagicMock(
            return_value=(
                np.array([[1.0, 2.0, 3.0]]),  # features
                np.array([85.0]),  # targets
                {"feature_columns": ["col1", "col2", "col3"]},  # metadata
            )
        )
        mock_fe_service.apply_one_hot_encoding = MagicMock(
            return_value={
                "encoding_applied": True,
                "features_encoded": ["tire_compound", "weather_condition"],
                "new_feature_count": 5,
                "encoding_metadata": {
                    "tire_compound": {"categories": 3, "type": "onehot"},
                    "weather_condition": {"categories": 2, "type": "onehot"},
                },
            }
        )

        service = SimulationService()

        request = DataProcessingRequest(
            session_key=12345,
            include_weather=True,
            include_grid=True,
            include_lap_times=True,
            include_pit_stops=True,
            processing_options={"apply_one_hot": True},
        )

        result = await service.apply_one_hot_encoding(request)

        assert result["encoding_applied"] is True
        assert result["features_encoded"] == ["tire_compound", "weather_condition"]
        assert result["new_feature_count"] == 5


@pytest.mark.asyncio
async def test_get_encoding_statistics_success():
    """Test successful encoding statistics retrieval."""
    with (
        patch("app.services.simulation_service.OpenF1Client") as mock_client_class,
        patch(
            "app.services.simulation_service.FeatureEngineeringService"
        ) as mock_fe_service_class,
    ):
        mock_client = mock_client_class.return_value
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)

        # Mock client methods for process_session_data
        mock_client.get_sessions = AsyncMock(
            return_value=[
                {
                    "session_key": 12345,
                    "session_name": "Race",
                    "track_name": "Monaco",
                    "country": "Monaco",
                    "year": 2024,
                }
            ]
        )
        mock_client.get_session_weather_summary = AsyncMock(return_value={})
        mock_client.get_session_grid_summary = AsyncMock(return_value={})
        mock_client.get_session_lap_times_summary = AsyncMock(return_value={})
        mock_client.get_session_pit_stops_summary = AsyncMock(return_value={})

        # Mock feature engineering service
        mock_fe_service = mock_fe_service_class.return_value
        mock_fe_service.fit_transform_features = MagicMock(
            return_value=(
                np.array([[1.0, 2.0, 3.0]]),  # features
                np.array([85.0]),  # targets
                {"feature_columns": ["col1", "col2", "col3"]},  # metadata
            )
        )
        mock_fe_service.get_encoding_statistics = MagicMock(
            return_value={
                "total_categorical_features": 5,
                "one_hot_encoded_features": 3,
                "label_encoded_features": 2,
                "total_encoded_features": 12,
                "encoding_distribution": {
                    "onehot": 3,
                    "label": 2,
                },
                "feature_cardinalities": {
                    "tire_compound": 3,
                    "weather_condition": 2,
                    "driver_team": 10,
                    "track_type": 3,
                    "lap_status": 2,
                },
            }
        )

        service = SimulationService()

        result = await service.get_encoding_statistics(12345)

        assert result["total_categorical_features"] == 5
        assert result["one_hot_encoded_features"] == 3
        assert result["label_encoded_features"] == 2
        assert result["total_encoded_features"] == 12
        assert result["encoding_distribution"]["onehot"] == 3


@pytest.mark.asyncio
async def test_get_simulation_result_cached():
    """Test getting simulation result from cache."""
    from app.api.v1.schemas import SimulationResponse
    from datetime import datetime, UTC

    service = SimulationService()

    # Add test simulation to cache
    test_simulation = SimulationResponse(
        simulation_id="cached_123",
        driver_id=1,
        track_id=1,
        season=2024,
        predicted_lap_time=85.0,
        confidence_score=0.9,
        weather_conditions="dry",
        car_setup={},
        created_at=datetime.now(UTC),
        processing_time_ms=1000,
    )
    service._simulation_cache["cached_123"] = test_simulation

    # Test getting cached result
    result = await service.get_simulation_result("cached_123")
    assert result.simulation_id == "cached_123"
    assert result.driver_id == 1
    assert result.predicted_lap_time == 85.0


@pytest.mark.asyncio
async def test_get_simulation_result_not_found():
    """Test getting simulation result when not found in cache."""
    from app.core.exceptions import InvalidSimulationParametersError

    service = SimulationService()

    with pytest.raises(
        InvalidSimulationParametersError, match="Simulation nonexistent_123 not found"
    ):
        await service.get_simulation_result("nonexistent_123")
