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
)


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
        from datetime import datetime

        mock_result = SimulationResponse(
            simulation_id="sim_123",
            driver_id=1,
            track_id=1,
            season=2024,
            predicted_lap_time=85.234,
            confidence_score=0.92,
            weather_conditions="dry",
            car_setup={},
            created_at=datetime.utcnow(),
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
    async def test_get_simulation_result_success(self):
        """Test successful retrieval of simulation result."""
        from datetime import datetime

        mock_result = SimulationResponse(
            simulation_id="sim_123",
            driver_id=1,
            track_id=1,
            season=2024,
            predicted_lap_time=85.234,
            confidence_score=0.92,
            weather_conditions="dry",
            car_setup={},
            created_at=datetime.utcnow(),
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
