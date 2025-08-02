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
        self.base_url = settings.openf1_api_url
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
        endpoint = "/drivers"
        params = {"year": season}

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
        endpoint = "/circuits"
        params = {"year": season}

        return await self._make_request("GET", endpoint, params=params)

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
        endpoint = "/lap_times"
        params = {"driver_id": driver_id, "circuit_id": track_id, "year": season}

        lap_times = await self._make_request("GET", endpoint, params=params)

        # Process the data to extract meaningful statistics
        return self._process_historical_data(lap_times)

    @alru_cache(maxsize=50)
    async def get_sessions(self, season: int) -> List[Dict]:
        """
        Get all sessions for a specific season.

        Args:
            season: F1 season year

        Returns:
            List of session data

        Raises:
            OpenF1APIError: If API call fails
        """
        # TODO: Replace with real API call when authentication is available
        # endpoint = "/v1/sessions"
        # params = {"year": season}
        # return await self._make_request("GET", endpoint, params=params)

        # Mock data for development
        mock_sessions = [
            {
                "session_key": 12345,
                "meeting_key": 1,
                "location": "Bahrain International Circuit",
                "session_type": "Race",
                "session_name": "2024 Bahrain Grand Prix",
                "date_start": "2024-03-02T15:00:00Z",
                "date_end": "2024-03-02T17:00:00Z",
                "country_name": "Bahrain",
                "circuit_short_name": "Bahrain International Circuit",
                "year": 2024,
            },
            {
                "session_key": 12346,
                "meeting_key": 1,
                "location": "Bahrain International Circuit",
                "session_type": "Qualifying",
                "session_name": "2024 Bahrain Grand Prix Qualifying",
                "date_start": "2024-03-01T18:00:00Z",
                "date_end": "2024-03-01T19:00:00Z",
                "country_name": "Bahrain",
                "circuit_short_name": "Bahrain International Circuit",
                "year": 2024,
            },
            {
                "session_key": 12347,
                "meeting_key": 2,
                "location": "Jeddah Corniche Circuit",
                "session_type": "Race",
                "session_name": "2024 Saudi Arabian Grand Prix",
                "date_start": "2024-03-09T17:00:00Z",
                "date_end": "2024-03-09T19:00:00Z",
                "country_name": "Saudi Arabia",
                "circuit_short_name": "Jeddah Corniche Circuit",
                "year": 2024,
            },
        ]

        return mock_sessions

    @alru_cache(maxsize=100)
    async def get_weather_data(self, session_key: int) -> List[Dict]:
        """
        Get weather data for a specific session.

        Args:
            session_key: Session identifier

        Returns:
            List of weather data points

        Raises:
            OpenF1APIError: If API call fails
        """
        endpoint = "/v1/weather"
        params = {"session_key": session_key}

        return await self._make_request("GET", endpoint, params=params)

    async def get_session_weather_summary(self, session_key: int) -> Dict:
        """
        Get weather summary for a specific session.

        Args:
            session_key: Session identifier

        Returns:
            Weather summary with averages and conditions

        Raises:
            OpenF1APIError: If API call fails
        """
        weather_data = await self.get_weather_data(session_key)

        if not weather_data:
            return {
                "session_key": session_key,
                "weather_condition": "unknown",
                "avg_air_temperature": None,
                "avg_track_temperature": None,
                "avg_humidity": None,
                "avg_pressure": None,
                "avg_wind_speed": None,
                "total_rainfall": None,
                "data_points": 0,
            }

        # Calculate averages
        air_temps: List[float] = []
        track_temps: List[float] = []
        humidities: List[float] = []
        pressures: List[float] = []
        wind_speeds: List[float] = []
        rainfalls: List[float] = []

        for w in weather_data:
            if w.get("air_temperature") is not None:
                air_temps.append(float(w["air_temperature"]))
            if w.get("track_temperature") is not None:
                track_temps.append(float(w["track_temperature"]))
            if w.get("humidity") is not None:
                humidities.append(float(w["humidity"]))
            if w.get("pressure") is not None:
                pressures.append(float(w["pressure"]))
            if w.get("wind_speed") is not None:
                wind_speeds.append(float(w["wind_speed"]))
            if w.get("rainfall") is not None:
                rainfalls.append(float(w["rainfall"]))

        # Determine weather condition based on rainfall
        total_rainfall = sum(rainfalls) if rainfalls else 0.0
        weather_condition = "wet" if total_rainfall > 0.1 else "dry"

        return {
            "session_key": session_key,
            "weather_condition": weather_condition,
            "avg_air_temperature": (
                sum(air_temps) / len(air_temps) if air_temps else None
            ),
            "avg_track_temperature": (
                sum(track_temps) / len(track_temps) if track_temps else None
            ),
            "avg_humidity": sum(humidities) / len(humidities) if humidities else None,
            "avg_pressure": sum(pressures) / len(pressures) if pressures else None,
            "avg_wind_speed": (
                sum(wind_speeds) / len(wind_speeds) if wind_speeds else None
            ),
            "total_rainfall": total_rainfall,
            "data_points": len(weather_data),
        }

    @alru_cache(maxsize=100)
    async def get_starting_grid(self, session_key: int) -> List[Dict]:
        """
        Get starting grid data for a specific session.

        Args:
            session_key: Session identifier

        Returns:
            List of grid position data

        Raises:
            OpenF1APIError: If API call fails
        """
        # TODO: Replace with real API call when authentication is available
        # endpoint = "/v1/starting_grid"
        # params = {"session_key": session_key}
        # return await self._make_request("GET", endpoint, params=params)

        # Mock data for development
        mock_grid_data = [
            {
                "position": 1,
                "driver_id": 1,
                "driver_name": "Max Verstappen",
                "driver_code": "VER",
                "team_name": "Red Bull Racing",
                "qualifying_time": 78.241,
                "qualifying_gap": 0.0,
                "qualifying_laps": 3,
            },
            {
                "position": 2,
                "driver_id": 2,
                "driver_name": "Lewis Hamilton",
                "driver_code": "HAM",
                "team_name": "Mercedes",
                "qualifying_time": 78.456,
                "qualifying_gap": 0.215,
                "qualifying_laps": 3,
            },
            {
                "position": 3,
                "driver_id": 3,
                "driver_name": "Charles Leclerc",
                "driver_code": "LEC",
                "team_name": "Ferrari",
                "qualifying_time": 78.567,
                "qualifying_gap": 0.326,
                "qualifying_laps": 3,
            },
            {
                "position": 4,
                "driver_id": 4,
                "driver_name": "Lando Norris",
                "driver_code": "NOR",
                "team_name": "McLaren",
                "qualifying_time": 78.789,
                "qualifying_gap": 0.548,
                "qualifying_laps": 3,
            },
            {
                "position": 5,
                "driver_id": 5,
                "driver_name": "Carlos Sainz",
                "driver_code": "SAI",
                "team_name": "Ferrari",
                "qualifying_time": 78.901,
                "qualifying_gap": 0.660,
                "qualifying_laps": 3,
            },
        ]

        return mock_grid_data

    @alru_cache(maxsize=50)
    async def get_qualifying_results(self, session_key: int) -> List[Dict]:
        """
        Get qualifying results for a specific session.

        Args:
            session_key: Session identifier

        Returns:
            List of qualifying result data

        Raises:
            OpenF1APIError: If API call fails
        """
        # TODO: Replace with real API call when authentication is available
        # endpoint = "/v1/qualifying"
        # params = {"session_key": session_key}
        # return await self._make_request("GET", endpoint, params=params)

        # Mock data for development
        mock_qualifying_data = [
            {
                "driver_id": 1,
                "driver_name": "Max Verstappen",
                "driver_code": "VER",
                "team_name": "Red Bull Racing",
                "q1_time": 78.241,
                "q2_time": 77.856,
                "q3_time": 77.123,
                "gap_to_pole": 0.0,
                "laps_completed": 3,
            },
            {
                "driver_id": 2,
                "driver_name": "Lewis Hamilton",
                "driver_code": "HAM",
                "team_name": "Mercedes",
                "q1_time": 78.456,
                "q2_time": 78.123,
                "q3_time": 77.456,
                "gap_to_pole": 0.333,
                "laps_completed": 3,
            },
            {
                "driver_id": 3,
                "driver_name": "Charles Leclerc",
                "driver_code": "LEC",
                "team_name": "Ferrari",
                "q1_time": 78.567,
                "q2_time": 78.234,
                "q3_time": 77.567,
                "gap_to_pole": 0.444,
                "laps_completed": 3,
            },
            {
                "driver_id": 4,
                "driver_name": "Lando Norris",
                "driver_code": "NOR",
                "team_name": "McLaren",
                "q1_time": 78.789,
                "q2_time": 78.456,
                "q3_time": 77.789,
                "gap_to_pole": 0.666,
                "laps_completed": 3,
            },
            {
                "driver_id": 5,
                "driver_name": "Carlos Sainz",
                "driver_code": "SAI",
                "team_name": "Ferrari",
                "q1_time": 78.901,
                "q2_time": 78.567,
                "q3_time": 77.901,
                "gap_to_pole": 0.778,
                "laps_completed": 3,
            },
        ]

        return mock_qualifying_data

    async def get_session_grid_summary(self, session_key: int) -> Dict:
        """
        Get grid summary statistics for a specific session.

        Args:
            session_key: Session identifier

        Returns:
            Grid summary with statistics

        Raises:
            OpenF1APIError: If API call fails
        """
        grid_data = await self.get_starting_grid(session_key)
        qualifying_data = await self.get_qualifying_results(session_key)

        if not grid_data:
            return {
                "session_key": session_key,
                "pole_position": None,
                "fastest_qualifying_time": None,
                "slowest_qualifying_time": None,
                "average_qualifying_time": None,
                "time_gap_pole_to_last": None,
                "teams_represented": [],
            }

        # Create a mapping of driver_id to qualifying data
        qualifying_map = {}
        for q in qualifying_data:
            driver_id = q.get("driver_id")
            if driver_id:
                qualifying_map[driver_id] = q

        # Process grid data and merge with qualifying data
        processed_grid = []
        teams = set()
        qualifying_times = []

        for position_data in grid_data:
            driver_id = position_data.get("driver_id")
            qualifying_info = qualifying_map.get(driver_id, {})

            # Extract team information
            team_name = position_data.get("team_name") or qualifying_info.get(
                "team_name"
            )
            if team_name:
                teams.add(team_name)

            # Extract qualifying time
            q_time = (
                qualifying_info.get("q1_time")
                or qualifying_info.get("q2_time")
                or qualifying_info.get("q3_time")
            )
            if q_time:
                try:
                    qualifying_times.append(float(q_time))
                except (ValueError, TypeError):
                    pass

            processed_position = {
                "position": position_data.get("position"),
                "driver_id": driver_id,
                "driver_name": position_data.get("driver_name")
                or qualifying_info.get("driver_name"),
                "driver_code": position_data.get("driver_code")
                or qualifying_info.get("driver_code"),
                "team_name": team_name,
                "qualifying_time": q_time,
                "qualifying_gap": qualifying_info.get("gap_to_pole"),
                "qualifying_laps": qualifying_info.get("laps_completed"),
            }
            processed_grid.append(processed_position)

        # Calculate statistics
        fastest_time = min(qualifying_times) if qualifying_times else None
        slowest_time = max(qualifying_times) if qualifying_times else None
        avg_time = (
            sum(qualifying_times) / len(qualifying_times) if qualifying_times else None
        )
        time_gap = (
            slowest_time - fastest_time if fastest_time and slowest_time else None
        )

        # Find pole position
        pole_position = None
        for pos in processed_grid:
            if pos.get("position") == 1:
                pole_position = pos
                break

        return {
            "session_key": session_key,
            "pole_position": pole_position,
            "fastest_qualifying_time": fastest_time,
            "slowest_qualifying_time": slowest_time,
            "average_qualifying_time": avg_time,
            "time_gap_pole_to_last": time_gap,
            "teams_represented": list(teams),
        }

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
            if "lap_time" in lap and lap["lap_time"]:
                try:
                    # Convert lap time string to seconds if needed
                    lap_time = lap["lap_time"]
                    if isinstance(lap_time, str):
                        # Parse time format like "1:23.456"
                        parts = lap_time.split(":")
                        if len(parts) == 2:
                            minutes = int(parts[0])
                            seconds = float(parts[1])
                            lap_time_seconds = minutes * 60 + seconds
                        else:
                            lap_time_seconds = float(lap_time)
                    else:
                        lap_time_seconds = float(lap_time)

                    times.append(lap_time_seconds)
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
