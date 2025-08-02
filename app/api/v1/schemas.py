"""
Pydantic schemas for API request/response validation.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class DriverResponse(BaseModel):
    """Response schema for driver information."""

    driver_id: int = Field(..., description="Unique driver identifier")
    name: str = Field(..., description="Driver's full name")
    code: str = Field(..., description="Driver's 3-letter code")
    team: str = Field(..., description="Driver's team name")
    nationality: str = Field(..., description="Driver's nationality")

    model_config = {
        "json_schema_extra": {
            "example": {
                "driver_id": 1,
                "name": "Max Verstappen",
                "code": "VER",
                "team": "Red Bull Racing",
                "nationality": "Dutch",
            }
        }
    }


class TrackResponse(BaseModel):
    """Response schema for track information."""

    track_id: int = Field(..., description="Unique track identifier")
    name: str = Field(..., description="Track name")
    country: str = Field(..., description="Country where the track is located")
    circuit_length: float = Field(..., description="Track length in kilometers")
    number_of_laps: int = Field(..., description="Number of laps in the race")

    model_config = {
        "json_schema_extra": {
            "example": {
                "track_id": 1,
                "name": "Silverstone Circuit",
                "country": "United Kingdom",
                "circuit_length": 5.891,
                "number_of_laps": 52,
            }
        }
    }


class SimulationRequest(BaseModel):
    """Request schema for simulation parameters."""

    driver_id: int = Field(..., description="Driver ID for the simulation")
    track_id: int = Field(..., description="Track ID for the simulation")
    season: int = Field(..., description="F1 season year", ge=1950, le=2030)
    weather_conditions: Optional[str] = Field(
        default="dry",
        description="Weather conditions (dry, wet, intermediate)",
        pattern="^(dry|wet|intermediate)$",
    )
    car_setup: Optional[dict] = Field(
        default_factory=dict, description="Car setup parameters (optional)"
    )

    @field_validator("season")
    @classmethod
    def validate_season(cls, v):
        """Validate that the season is reasonable."""
        if v < 1950 or v > 2030:
            raise ValueError("Season must be between 1950 and 2030")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "driver_id": 1,
                "track_id": 1,
                "season": 2024,
                "weather_conditions": "dry",
                "car_setup": {"downforce": "high", "tire_compound": "soft"},
            }
        }
    }


class SimulationResponse(BaseModel):
    """Response schema for simulation results."""

    simulation_id: str = Field(..., description="Unique simulation identifier")
    driver_id: int = Field(..., description="Driver ID used in simulation")
    track_id: int = Field(..., description="Track ID used in simulation")
    season: int = Field(..., description="Season used in simulation")
    predicted_lap_time: float = Field(..., description="Predicted lap time in seconds")
    confidence_score: float = Field(..., description="Confidence score (0-1)")
    weather_conditions: str = Field(..., description="Weather conditions used")
    car_setup: dict = Field(..., description="Car setup parameters used")
    created_at: datetime = Field(..., description="Simulation creation timestamp")
    processing_time_ms: int = Field(
        ..., description="Time taken to process simulation in milliseconds"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "simulation_id": "sim_123456789",
                "driver_id": 1,
                "track_id": 1,
                "season": 2024,
                "predicted_lap_time": 78.456,
                "confidence_score": 0.85,
                "weather_conditions": "dry",
                "car_setup": {"downforce": "high", "tire_compound": "soft"},
                "created_at": "2024-01-15T10:30:00Z",
                "processing_time_ms": 1250,
            }
        }
    }


class SessionResponse(BaseModel):
    """Response schema for session information."""

    session_key: int = Field(..., description="Unique session identifier")
    meeting_key: int = Field(..., description="Meeting identifier")
    location: str = Field(..., description="Track location")
    session_type: str = Field(
        ..., description="Session type (Practice, Qualifying, Race)"
    )
    session_name: str = Field(..., description="Session name")
    date_start: datetime = Field(..., description="Session start time")
    date_end: datetime = Field(..., description="Session end time")
    country_name: str = Field(..., description="Country name")
    circuit_short_name: str = Field(..., description="Circuit short name")
    year: int = Field(..., description="Season year")

    model_config = {
        "json_schema_extra": {
            "example": {
                "session_key": 9472,
                "meeting_key": 1229,
                "location": "Sakhir",
                "session_type": "Race",
                "session_name": "Race",
                "date_start": "2024-03-02T15:00:00+00:00",
                "date_end": "2024-03-02T17:00:00+00:00",
                "country_name": "Bahrain",
                "circuit_short_name": "Sakhir",
                "year": 2024,
            }
        }
    }


class WeatherDataResponse(BaseModel):
    """Response schema for weather data point."""

    date: datetime = Field(..., description="Weather measurement timestamp")
    session_key: int = Field(..., description="Session identifier")
    air_temperature: float = Field(..., description="Air temperature in Celsius")
    track_temperature: float = Field(..., description="Track temperature in Celsius")
    humidity: float = Field(..., description="Humidity percentage")
    pressure: float = Field(..., description="Atmospheric pressure in hPa")
    wind_speed: float = Field(..., description="Wind speed in m/s")
    wind_direction: int = Field(..., description="Wind direction in degrees")
    rainfall: float = Field(..., description="Rainfall in mm")

    model_config = {
        "json_schema_extra": {
            "example": {
                "date": "2024-03-02T14:03:56.523000+00:00",
                "session_key": 9472,
                "air_temperature": 18.9,
                "track_temperature": 26.5,
                "humidity": 46.0,
                "pressure": 1017.1,
                "wind_speed": 0.9,
                "wind_direction": 162,
                "rainfall": 0.0,
            }
        }
    }


class WeatherSummaryResponse(BaseModel):
    """Response schema for weather summary."""

    session_key: int = Field(..., description="Session identifier")
    weather_condition: str = Field(
        ..., description="Weather condition (dry/wet/unknown)"
    )
    avg_air_temperature: Optional[float] = Field(
        None, description="Average air temperature in Celsius"
    )
    avg_track_temperature: Optional[float] = Field(
        None, description="Average track temperature in Celsius"
    )
    avg_humidity: Optional[float] = Field(
        None, description="Average humidity percentage"
    )
    avg_pressure: Optional[float] = Field(
        None, description="Average atmospheric pressure in hPa"
    )
    avg_wind_speed: Optional[float] = Field(
        None, description="Average wind speed in m/s"
    )
    total_rainfall: float = Field(..., description="Total rainfall in mm")
    data_points: int = Field(..., description="Number of weather data points")

    model_config = {
        "json_schema_extra": {
            "example": {
                "session_key": 9472,
                "weather_condition": "dry",
                "avg_air_temperature": 18.5,
                "avg_track_temperature": 24.2,
                "avg_humidity": 48.5,
                "avg_pressure": 1017.2,
                "avg_wind_speed": 0.8,
                "total_rainfall": 0.0,
                "data_points": 120,
            }
        }
    }


class ErrorResponse(BaseModel):
    """Standard error response schema."""

    detail: str = Field(..., description="Human-readable error message")
    code: str = Field(..., description="Error code for programmatic handling")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="Error timestamp"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "detail": "Driver with ID 999 not found",
                "code": "DRIVER_NOT_FOUND",
                "timestamp": "2024-01-15T10:30:00Z",
            }
        }
    }
