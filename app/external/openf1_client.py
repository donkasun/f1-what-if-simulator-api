"""
OpenF1 API client for fetching F1 data.
"""

import time
from typing import Dict, List, Optional

import httpx
import structlog
from async_lru import alru_cache

from app.core.config import settings
from app.core.exceptions import OpenF1APIError

logger = structlog.get_logger()


class OpenF1Client:
    """Async client for OpenF1 API with caching and error handling."""

    def __init__(self):
        """Initialize the OpenF1 client."""
        self.base_url = "https://api.openf1.org/v1"
        self.timeout = settings.openf1_api_timeout
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        """Async context manager entry."""
        await self._ensure_client()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    async def _ensure_client(self) -> None:
        """Ensure the HTTP client is initialized."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
            )

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    @alru_cache(maxsize=100)
    async def get_drivers(self, season: int) -> List[Dict]:
        """
        Get all drivers for a specific season.

        Args:
            season: F1 season year

        Returns:
            List of driver data

        Raises:
            OpenF1APIError: If API call fails
        """
        meetings = await self.get_meetings(season)
        all_drivers = {}
        for meeting in meetings:
            meeting_key = meeting.get("meeting_key")
            if meeting_key:
                sessions = await self.get_sessions(meeting_key)
                for session in sessions:
                    session_key = session.get("session_key")
                    if session_key:
                        endpoint = "/drivers"
                        params = {"session_key": session_key}
                        drivers_data = await self._make_request(
                            "GET", endpoint, params=params
                        )
                        for driver in drivers_data:
                            all_drivers[driver.get("driver_number")] = driver

        return list(all_drivers.values())

    @alru_cache(maxsize=50)
    async def get_meetings(self, season: int) -> List[Dict]:
        """
        Get all meetings for a specific season.

        Args:
            season: F1 season year

        Returns:
            List of meeting data

        Raises:
            OpenF1APIError: If API call fails
        """
        endpoint = "/meetings"
        params = {"year": season}

        return await self._make_request("GET", endpoint, params=params)

    @alru_cache(maxsize=50)
    async def get_sessions(self, meeting_key: int) -> List[Dict]:
        """
        Get all sessions for a specific meeting.

        Args:
            meeting_key: The unique identifier for the meeting.

        Returns:
            List of session data

        Raises:
            OpenF1APIError: If API call fails
        """
        endpoint = "/sessions"
        params = {"meeting_key": meeting_key}

        return await self._make_request("GET", endpoint, params=params)

    @alru_cache(maxsize=50)
    async def get_tracks(self, season: int) -> List[Dict]:
        """
        Get all tracks for a specific season.

        Args:
            season: F1 season year

        Returns:
            List of track data

        Raises:
            OpenF1APIError: If API call fails
        """
        meetings_data = await self.get_meetings(season)

        tracks = []
        for meeting in meetings_data:
            tracks.append(
                {
                    "track_id": meeting.get("circuit_key"),
                    "name": meeting.get("circuit_short_name"),
                    "country": meeting.get("country_name"),
                    "circuit_length": None,  # Not available in this endpoint
                    "number_of_laps": None,  # Not available in this endpoint
                }
            )
        return tracks

    async def get_historical_data(
        self, driver_id: int, track_id: int, season: int
    ) -> Dict:
        """
        Get historical performance data for a driver at a specific track.

        Args:
            driver_id: Driver identifier
            track_id: Track identifier
            season: F1 season year

        Returns:
            Historical performance data

        Raises:
            OpenF1APIError: If API call fails
        """
        meetings = await self.get_meetings(season)
        session_key = None
        for meeting in meetings:
            if meeting.get("circuit_key") == track_id:
                sessions = await self.get_sessions(meeting.get("meeting_key"))
                for session in sessions:
                    if session.get("session_name") == "Race":
                        session_key = session.get("session_key")
                        break
            if session_key:
                break

        if not session_key:
            raise OpenF1APIError(
                f"No race session found for track {track_id} in {season}"
            )

        endpoint = "/laps"
        params = {"session_key": session_key, "driver_number": driver_id}

        lap_times = await self._make_request("GET", endpoint, params=params)

        # Process the data to extract meaningful statistics
        return self._process_historical_data(lap_times)

    async def _make_request(
        self, method: str, endpoint: str, params: Optional[Dict] = None
    ) -> List[Dict]:  # type: ignore
        """
        Make an HTTP request to the OpenF1 API.

        Args:
            method: HTTP method
            endpoint: API endpoint
            params: Query parameters

        Returns:
            API response data

        Raises:
            OpenF1APIError: If request fails
        """
        await self._ensure_client()

        url = f"{self.base_url}{endpoint}"
        start_time = time.time()

        try:
            logger.info(
                "Making OpenF1 API request",
                method=method,
                url=url,
                params=params,
            )

            if self._client is None:
                raise OpenF1APIError("HTTP client not initialized")

            time.sleep(1)  # Add a delay to avoid rate limiting
            response = await self._client.request(method, url, params=params)
            response_time_ms = int((time.time() - start_time) * 1000)

            if response.status_code == 200:
                data = response.json()
                logger.info(
                    "OpenF1 API request successful",
                    method=method,
                    url=url,
                    status_code=response.status_code,
                    response_time_ms=response_time_ms,
                    data_count=len(data) if isinstance(data, list) else 1,
                )
                return data
            else:
                logger.error(
                    "OpenF1 API request failed",
                    method=method,
                    url=url,
                    status_code=response.status_code,
                    response_time_ms=response_time_ms,
                    response_text=response.text,
                )
                raise OpenF1APIError(
                    f"OpenF1 API request failed with status {response.status_code}",
                    response.status_code,
                )

        except httpx.TimeoutException as e:
            response_time_ms = int((time.time() - start_time) * 1000)
            logger.error(
                "OpenF1 API request timeout",
                method=method,
                url=url,
                response_time_ms=response_time_ms,
                error=str(e),
            )
            raise OpenF1APIError(f"OpenF1 API request timeout: {str(e)}", 408)

        except httpx.RequestError as e:
            response_time_ms = int((time.time() - start_time) * 1000)
            logger.error(
                "OpenF1 API request error",
                method=method,
                url=url,
                response_time_ms=response_time_ms,
                error=str(e),
            )
            raise OpenF1APIError(f"OpenF1 API request error: {str(e)}", 500)

        except Exception as e:
            response_time_ms = int((time.time() - start_time) * 1000)
            logger.error(
                "Unexpected error in OpenF1 API request",
                method=method,
                url=url,
                response_time_ms=response_time_ms,
                error=str(e),
                exc_info=True,
            )
            raise OpenF1APIError(f"Unexpected error: {str(e)}", 500)

    def _process_historical_data(self, lap_times: List[Dict]) -> Dict:
        """
        Process raw lap time data into meaningful statistics.

        Args:
            lap_times: Raw lap time data from API

        Returns:
            Processed historical data with statistics
        """
        if not lap_times:
            return {
                "avg_lap_time": 0.0,
                "best_lap_time": 0.0,
                "consistency_score": 0.0,
                "data_points": 0,
            }

        # Extract lap times (assuming they're in seconds)
        times = []
        for lap in lap_times:
            if "lap_duration" in lap and lap["lap_duration"]:
                try:
                    times.append(float(lap["lap_duration"]))
                except (ValueError, TypeError):
                    continue

        if not times:
            return {
                "avg_lap_time": 0.0,
                "best_lap_time": 0.0,
                "consistency_score": 0.0,
                "data_points": 0,
            }

        # Calculate statistics
        avg_lap_time = sum(times) / len(times)
        best_lap_time = min(times)

        # Calculate consistency score (lower standard deviation = higher consistency)
        variance = sum((t - avg_lap_time) ** 2 for t in times) / len(times)
        std_dev = variance**0.5
        consistency_score = max(0.0, 1.0 - (std_dev / avg_lap_time))

        return {
            "avg_lap_time": avg_lap_time,
            "best_lap_time": best_lap_time,
            "consistency_score": consistency_score,
            "data_points": len(times),
        }
