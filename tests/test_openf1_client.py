"""
Tests for OpenF1 client.
"""

import pytest
from unittest.mock import patch, MagicMock
from httpx import HTTPStatusError
from app.external.openf1_client import OpenF1Client, OpenF1APIError


class TestOpenF1ClientInitialization:
    """Test cases for OpenF1 client initialization."""

    def test_client_initialization(self):
        """Test that the client can be initialized."""
        client = OpenF1Client()
        assert client is not None
        assert client.base_url == "https://api.openf1.org"
        assert client._client is None

    @pytest.mark.asyncio
    async def test_client_context_manager(self):
        """Test client as context manager."""
        async with OpenF1Client() as client:
            assert client._client is not None
            assert hasattr(client._client, "aclose")

    @pytest.mark.asyncio
    async def test_client_manual_close(self):
        """Test manual client close."""
        client = OpenF1Client()
        await client._ensure_client()
        assert client._client is not None

        await client.close()
        assert client._client is None


class TestOpenF1ClientMakeRequest:
    """Test cases for _make_request method."""

    @pytest.mark.asyncio
    async def test_make_request_success(self):
        """Test successful API request."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"data": "test"}

        with patch("httpx.AsyncClient.request", return_value=mock_response):
            async with OpenF1Client() as client:
                result = await client._make_request("GET", "/test")

                assert result == {"data": "test"}

    @pytest.mark.asyncio
    async def test_make_request_http_error(self):
        """Test API request with HTTP error."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.raise_for_status.side_effect = HTTPStatusError(
            "Not Found", request=MagicMock(), response=mock_response
        )

        with patch("httpx.AsyncClient.request", return_value=mock_response):
            async with OpenF1Client() as client:
                with pytest.raises(OpenF1APIError, match="API request failed"):
                    await client._make_request("GET", "/test")

    @pytest.mark.asyncio
    async def test_make_request_connection_error(self):
        """Test API request with connection error."""
        with patch(
            "httpx.AsyncClient.request", side_effect=Exception("Connection error")
        ):
            async with OpenF1Client() as client:
                with pytest.raises(OpenF1APIError, match="Unexpected error"):
                    await client._make_request("GET", "/test")

    @pytest.mark.asyncio
    async def test_make_request_uninitialized_client(self):
        """Test request with uninitialized client."""
        client = OpenF1Client()
        client._client = None

        # Mock _ensure_client to not initialize the client
        with patch.object(client, "_ensure_client"):
            with pytest.raises(OpenF1APIError, match="HTTP client not initialized"):
                await client._make_request("GET", "/test")


class TestOpenF1ClientGetSessions:
    """Test cases for get_sessions method."""

    @pytest.mark.asyncio
    async def test_get_sessions_success(self):
        """Test successful session retrieval."""
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

        with patch.object(OpenF1Client, "_make_request", return_value=mock_sessions):
            async with OpenF1Client() as client:
                result = await client.get_sessions(2024)

                assert result == mock_sessions
                assert len(result) == 1
                assert result[0]["session_key"] == 12345

    @pytest.mark.asyncio
    async def test_get_sessions_caching(self):
        """Test that sessions are cached."""
        mock_sessions = [{"session_key": 12345, "year": 2024}]

        with patch.object(OpenF1Client, "_make_request", return_value=mock_sessions):
            async with OpenF1Client() as client:
                # First call
                result1 = await client.get_sessions(2024)

                # Second call should use cache
                result2 = await client.get_sessions(2024)

                assert result1 == result2
                # Verify _make_request was only called once
                assert client._make_request.call_count == 1

    @pytest.mark.asyncio
    async def test_get_sessions_api_error(self):
        """Test session retrieval with API error."""
        with patch.object(
            OpenF1Client, "_make_request", side_effect=OpenF1APIError("API error")
        ):
            async with OpenF1Client() as client:
                with pytest.raises(OpenF1APIError, match="API error"):
                    await client.get_sessions(2024)


class TestOpenF1ClientGetWeatherData:
    """Test cases for get_weather_data method."""

    @pytest.mark.asyncio
    async def test_get_weather_data_success(self):
        """Test successful weather data retrieval."""
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

        with patch.object(
            OpenF1Client, "_make_request", return_value=mock_weather_data
        ):
            async with OpenF1Client() as client:
                result = await client.get_weather_data(12345)

                assert result == mock_weather_data
                assert len(result) == 1
                assert result[0]["session_key"] == 12345
                assert result[0]["air_temperature"] == 25.5

    @pytest.mark.asyncio
    async def test_get_weather_data_caching(self):
        """Test that weather data is cached."""
        mock_weather_data = [{"session_key": 12345, "air_temperature": 25.5}]

        with patch.object(
            OpenF1Client, "_make_request", return_value=mock_weather_data
        ):
            async with OpenF1Client() as client:
                # First call
                result1 = await client.get_weather_data(12345)

                # Second call should use cache
                result2 = await client.get_weather_data(12345)

                assert result1 == result2
                # Verify _make_request was only called once
                assert client._make_request.call_count == 1

    @pytest.mark.asyncio
    async def test_get_weather_data_api_error(self):
        """Test weather data retrieval with API error."""
        with patch.object(
            OpenF1Client, "_make_request", side_effect=OpenF1APIError("API error")
        ):
            async with OpenF1Client() as client:
                with pytest.raises(OpenF1APIError, match="API error"):
                    await client.get_weather_data(12345)


class TestOpenF1ClientGetSessionWeatherSummary:
    """Test cases for get_session_weather_summary method."""

    @pytest.mark.asyncio
    async def test_get_session_weather_summary_success(self):
        """Test successful weather summary calculation."""
        mock_weather_data = [
            {
                "air_temperature": 25.0,
                "track_temperature": 35.0,
                "humidity": 60.0,
                "pressure": 1013.0,
                "wind_speed": 5.0,
                "rainfall": 0.0,
            },
            {
                "air_temperature": 26.0,
                "track_temperature": 36.0,
                "humidity": 70.0,
                "pressure": 1014.0,
                "wind_speed": 6.0,
                "rainfall": 0.0,
            },
        ]

        with patch.object(
            OpenF1Client, "get_weather_data", return_value=mock_weather_data
        ):
            async with OpenF1Client() as client:
                result = await client.get_session_weather_summary(12345)

                assert result["session_key"] == 12345
                assert result["weather_condition"] == "dry"
                assert result["avg_air_temperature"] == 25.5
                assert result["avg_track_temperature"] == 35.5
                assert result["avg_humidity"] == 65.0
                assert result["avg_pressure"] == 1013.5
                assert result["avg_wind_speed"] == 5.5
                assert result["total_rainfall"] == 0.0
                assert result["data_points"] == 2

    @pytest.mark.asyncio
    async def test_get_session_weather_summary_with_missing_data(self):
        """Test weather summary with missing data points."""
        mock_weather_data = [
            {
                "air_temperature": 25.0,
                "track_temperature": None,
                "humidity": 60.0,
                "pressure": None,
                "wind_speed": 5.0,
                "rainfall": 0.0,
            },
            {
                "air_temperature": None,
                "track_temperature": 36.0,
                "humidity": None,
                "pressure": 1014.0,
                "wind_speed": None,
                "rainfall": 0.0,
            },
        ]

        with patch.object(
            OpenF1Client, "get_weather_data", return_value=mock_weather_data
        ):
            async with OpenF1Client() as client:
                result = await client.get_session_weather_summary(12345)

                assert result["session_key"] == 12345
                assert result["weather_condition"] == "dry"
                assert result["avg_air_temperature"] == 25.0
                assert result["avg_track_temperature"] == 36.0
                assert result["avg_humidity"] == 60.0
                assert result["avg_pressure"] == 1014.0
                assert result["avg_wind_speed"] == 5.0
                assert result["total_rainfall"] == 0.0
                assert result["data_points"] == 2

    @pytest.mark.asyncio
    async def test_get_session_weather_summary_wet_conditions(self):
        """Test weather summary for wet conditions."""
        mock_weather_data = [
            {
                "air_temperature": 20.0,
                "track_temperature": 25.0,
                "humidity": 85.0,
                "pressure": 1000.0,
                "wind_speed": 10.0,
                "rainfall": 5.0,
            }
        ]

        with patch.object(
            OpenF1Client, "get_weather_data", return_value=mock_weather_data
        ):
            async with OpenF1Client() as client:
                result = await client.get_session_weather_summary(12345)

                assert result["weather_condition"] == "wet"
                assert result["total_rainfall"] == 5.0

    @pytest.mark.asyncio
    async def test_get_session_weather_summary_empty_data(self):
        """Test weather summary with empty data."""
        with patch.object(OpenF1Client, "get_weather_data", return_value=[]):
            async with OpenF1Client() as client:
                result = await client.get_session_weather_summary(12345)

                assert result["session_key"] == 12345
                assert result["weather_condition"] == "unknown"
                assert result["data_points"] == 0
                assert result["avg_air_temperature"] is None
                assert result["avg_track_temperature"] is None

    @pytest.mark.asyncio
    async def test_get_session_weather_summary_api_error(self):
        """Test weather summary with API error."""
        with patch.object(
            OpenF1Client, "get_weather_data", side_effect=OpenF1APIError("API error")
        ):
            async with OpenF1Client() as client:
                with pytest.raises(OpenF1APIError, match="API error"):
                    await client.get_session_weather_summary(12345)


class TestOpenF1ClientErrorHandling:
    """Test cases for error handling."""

    @pytest.mark.asyncio
    async def test_client_timeout_handling(self):
        """Test timeout handling."""
        with patch("httpx.AsyncClient.request", side_effect=Exception("Timeout")):
            async with OpenF1Client() as client:
                with pytest.raises(OpenF1APIError, match="Unexpected error"):
                    await client._make_request("GET", "/test")

    @pytest.mark.asyncio
    async def test_client_json_parsing_error(self):
        """Test JSON parsing error handling."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.side_effect = Exception("JSON parsing error")

        with patch("httpx.AsyncClient.request", return_value=mock_response):
            async with OpenF1Client() as client:
                with pytest.raises(OpenF1APIError, match="Unexpected error"):
                    await client._make_request("GET", "/test")
