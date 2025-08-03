"""
Pydantic schemas for API request/response validation.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any

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
    circuit_length: Optional[float] = Field(
        None, description="Track length in kilometers"
    )
    number_of_laps: Optional[int] = Field(
        None, description="Number of laps in the race"
    )

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
                "detail": "Invalid simulation parameters provided",
                "code": "INVALID_PARAMETERS",
                "timestamp": "2024-01-15T10:30:00Z",
            }
        }
    }


class GridPositionResponse(BaseModel):
    """Response schema for individual grid position."""

    position: int = Field(..., description="Starting grid position")
    driver_id: int = Field(..., description="Driver identifier")
    driver_name: str = Field(..., description="Driver's full name")
    driver_code: str = Field(..., description="Driver's 3-letter code")
    team_name: str = Field(..., description="Team name")
    qualifying_time: Optional[float] = Field(
        None, description="Qualifying time in seconds"
    )
    qualifying_gap: Optional[float] = Field(
        None, description="Gap to pole position in seconds"
    )
    qualifying_laps: Optional[int] = Field(
        None, description="Number of qualifying laps completed"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "position": 1,
                "driver_id": 1,
                "driver_name": "Max Verstappen",
                "driver_code": "VER",
                "team_name": "Red Bull Racing",
                "qualifying_time": 78.241,
                "qualifying_gap": 0.0,
                "qualifying_laps": 3,
            }
        }
    }


class StartingGridResponse(BaseModel):
    """Response schema for complete starting grid."""

    session_key: int = Field(..., description="Session identifier")
    session_name: str = Field(..., description="Session name")
    track_name: str = Field(..., description="Track name")
    country: str = Field(..., description="Country")
    year: int = Field(..., description="Season year")
    total_drivers: int = Field(..., description="Total number of drivers")
    grid_positions: List[GridPositionResponse] = Field(
        ..., description="List of grid positions"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "session_key": 12345,
                "session_name": "2024 Bahrain Grand Prix",
                "track_name": "Bahrain International Circuit",
                "country": "Bahrain",
                "year": 2024,
                "total_drivers": 20,
                "grid_positions": [
                    {
                        "position": 1,
                        "driver_id": 1,
                        "driver_name": "Max Verstappen",
                        "driver_code": "VER",
                        "team_name": "Red Bull Racing",
                        "qualifying_time": 78.241,
                        "qualifying_gap": 0.0,
                        "qualifying_laps": 3,
                    }
                ],
            }
        }
    }


class GridSummaryResponse(BaseModel):
    """Response schema for grid summary statistics."""

    session_key: int = Field(..., description="Session identifier")
    pole_position: Optional[GridPositionResponse] = Field(
        None, description="Pole position driver"
    )
    fastest_qualifying_time: Optional[float] = Field(
        None, description="Fastest qualifying time in seconds"
    )
    slowest_qualifying_time: Optional[float] = Field(
        None, description="Slowest qualifying time in seconds"
    )
    average_qualifying_time: Optional[float] = Field(
        None, description="Average qualifying time in seconds"
    )
    time_gap_pole_to_last: Optional[float] = Field(
        None, description="Time gap from pole to last position in seconds"
    )
    teams_represented: List[str] = Field(..., description="List of teams in the grid")

    model_config = {
        "json_schema_extra": {
            "example": {
                "session_key": 12345,
                "pole_position": {
                    "position": 1,
                    "driver_id": 1,
                    "driver_name": "Max Verstappen",
                    "driver_code": "VER",
                    "team_name": "Red Bull Racing",
                    "qualifying_time": 78.241,
                    "qualifying_gap": 0.0,
                    "qualifying_laps": 3,
                },
                "fastest_qualifying_time": 78.241,
                "slowest_qualifying_time": 82.156,
                "average_qualifying_time": 80.198,
                "time_gap_pole_to_last": 3.915,
                "teams_represented": ["Red Bull Racing", "Mercedes", "Ferrari"],
            }
        }
    }


class LapTimeResponse(BaseModel):
    """Response schema for individual lap time data."""

    lap_number: int = Field(..., description="Lap number")
    driver_id: int = Field(..., description="Driver identifier")
    driver_name: str = Field(..., description="Driver's full name")
    driver_code: str = Field(..., description="Driver's 3-letter code")
    team_name: str = Field(..., description="Team name")
    lap_time: Optional[float] = Field(None, description="Lap time in seconds")
    sector_1_time: Optional[float] = Field(None, description="Sector 1 time in seconds")
    sector_2_time: Optional[float] = Field(None, description="Sector 2 time in seconds")
    sector_3_time: Optional[float] = Field(None, description="Sector 3 time in seconds")
    tire_compound: Optional[str] = Field(None, description="Tire compound used")
    fuel_load: Optional[float] = Field(None, description="Fuel load in kg")
    lap_status: str = Field(..., description="Lap status (valid/invalid/dnf)")
    timestamp: datetime = Field(..., description="Lap timestamp")

    model_config = {
        "json_schema_extra": {
            "example": {
                "lap_number": 15,
                "driver_id": 1,
                "driver_name": "Max Verstappen",
                "driver_code": "VER",
                "team_name": "Red Bull Racing",
                "lap_time": 78.456,
                "sector_1_time": 25.123,
                "sector_2_time": 26.789,
                "sector_3_time": 26.544,
                "tire_compound": "soft",
                "fuel_load": 45.2,
                "lap_status": "valid",
                "timestamp": "2024-03-02T15:30:00Z",
            }
        }
    }


class PitStopResponse(BaseModel):
    """Response schema for individual pit stop data."""

    pit_stop_number: int = Field(..., description="Pit stop number in the race")
    driver_id: int = Field(..., description="Driver identifier")
    driver_name: str = Field(..., description="Driver's full name")
    driver_code: str = Field(..., description="Driver's 3-letter code")
    team_name: str = Field(..., description="Team name")
    lap_number: int = Field(..., description="Lap when pit stop occurred")
    pit_duration: float = Field(..., description="Pit stop duration in seconds")
    tire_compound_in: str = Field(..., description="Tire compound going in")
    tire_compound_out: str = Field(..., description="Tire compound coming out")
    fuel_added: Optional[float] = Field(None, description="Fuel added in kg")
    pit_reason: str = Field(..., description="Reason for pit stop")
    timestamp: datetime = Field(..., description="Pit stop timestamp")

    model_config = {
        "json_schema_extra": {
            "example": {
                "pit_stop_number": 1,
                "driver_id": 1,
                "driver_name": "Max Verstappen",
                "driver_code": "VER",
                "team_name": "Red Bull Racing",
                "lap_number": 18,
                "pit_duration": 2.8,
                "tire_compound_in": "medium",
                "tire_compound_out": "soft",
                "fuel_added": 15.5,
                "pit_reason": "tire_change",
                "timestamp": "2024-03-02T15:45:00Z",
            }
        }
    }


class DriverPerformanceResponse(BaseModel):
    """Response schema for driver performance metrics."""

    driver_id: int = Field(..., description="Driver identifier")
    driver_name: str = Field(..., description="Driver's full name")
    driver_code: str = Field(..., description="Driver's 3-letter code")
    team_name: str = Field(..., description="Team name")
    total_laps: int = Field(..., description="Total laps completed")
    best_lap_time: Optional[float] = Field(None, description="Best lap time in seconds")
    avg_lap_time: Optional[float] = Field(
        None, description="Average lap time in seconds"
    )
    consistency_score: float = Field(..., description="Lap consistency score (0-1)")
    total_pit_stops: int = Field(..., description="Total number of pit stops")
    total_pit_time: float = Field(
        ..., description="Total time spent in pits in seconds"
    )
    avg_pit_time: float = Field(..., description="Average pit stop time in seconds")
    tire_compounds_used: List[str] = Field(
        ..., description="List of tire compounds used"
    )
    final_position: Optional[int] = Field(None, description="Final race position")
    race_status: str = Field(..., description="Final race status (finished/dnf/dsq)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "driver_id": 1,
                "driver_name": "Max Verstappen",
                "driver_code": "VER",
                "team_name": "Red Bull Racing",
                "total_laps": 52,
                "best_lap_time": 77.123,
                "avg_lap_time": 78.456,
                "consistency_score": 0.92,
                "total_pit_stops": 2,
                "total_pit_time": 5.6,
                "avg_pit_time": 2.8,
                "tire_compounds_used": ["soft", "medium", "soft"],
                "final_position": 1,
                "race_status": "finished",
            }
        }
    }


class LapTimesResponse(BaseModel):
    """Response schema for complete lap times data."""

    session_key: int = Field(..., description="Session identifier")
    session_name: str = Field(..., description="Session name")
    track_name: str = Field(..., description="Track name")
    country: str = Field(..., description="Country")
    year: int = Field(..., description="Season year")
    total_laps: int = Field(..., description="Total laps in the session")
    lap_times: List[LapTimeResponse] = Field(..., description="List of lap times")

    model_config = {
        "json_schema_extra": {
            "example": {
                "session_key": 9472,
                "session_name": "2024 Bahrain Grand Prix",
                "track_name": "Bahrain International Circuit",
                "country": "Bahrain",
                "year": 2024,
                "total_laps": 52,
                "lap_times": [],
            }
        }
    }


class PitStopsResponse(BaseModel):
    """Response schema for complete pit stops data."""

    session_key: int = Field(..., description="Session identifier")
    session_name: str = Field(..., description="Session name")
    track_name: str = Field(..., description="Track name")
    country: str = Field(..., description="Country")
    year: int = Field(..., description="Season year")
    total_pit_stops: int = Field(..., description="Total pit stops in the session")
    pit_stops: List[PitStopResponse] = Field(..., description="List of pit stops")

    model_config = {
        "json_schema_extra": {
            "example": {
                "session_key": 9472,
                "session_name": "2024 Bahrain Grand Prix",
                "track_name": "Bahrain International Circuit",
                "country": "Bahrain",
                "year": 2024,
                "total_pit_stops": 15,
                "pit_stops": [],
            }
        }
    }


class DriverPerformanceSummaryResponse(BaseModel):
    """Response schema for driver performance summary."""

    session_key: int = Field(..., description="Session identifier")
    session_name: str = Field(..., description="Session name")
    track_name: str = Field(..., description="Track name")
    country: str = Field(..., description="Country")
    year: int = Field(..., description="Season year")
    total_drivers: int = Field(..., description="Total number of drivers")
    driver_performances: List[DriverPerformanceResponse] = Field(
        ..., description="List of driver performances"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "session_key": 9472,
                "session_name": "2024 Bahrain Grand Prix",
                "track_name": "Bahrain International Circuit",
                "country": "Bahrain",
                "year": 2024,
                "total_drivers": 20,
                "driver_performances": [],
            }
        }
    }


class DataProcessingRequest(BaseModel):
    """Request schema for data processing operations."""

    session_key: int = Field(..., description="Session identifier")
    include_weather: bool = Field(
        default=True, description="Include weather data in processing"
    )
    include_grid: bool = Field(
        default=True, description="Include grid data in processing"
    )
    include_lap_times: bool = Field(
        default=True, description="Include lap times data in processing"
    )
    include_pit_stops: bool = Field(
        default=True, description="Include pit stops data in processing"
    )
    processing_options: Optional[dict] = Field(
        default_factory=dict, description="Additional processing options"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "session_key": 9472,
                "include_weather": True,
                "include_grid": True,
                "include_lap_times": True,
                "include_pit_stops": True,
                "processing_options": {
                    "align_timestamps": True,
                    "fill_missing_values": True,
                    "normalize_features": True,
                },
            }
        }
    }


class ProcessedDataPoint(BaseModel):
    """Schema for a single processed data point."""

    timestamp: datetime = Field(..., description="Data point timestamp")
    driver_id: int = Field(..., description="Driver identifier")
    lap_number: int = Field(..., description="Lap number")
    lap_time: Optional[float] = Field(None, description="Lap time in seconds")
    sector_1_time: Optional[float] = Field(None, description="Sector 1 time in seconds")
    sector_2_time: Optional[float] = Field(None, description="Sector 2 time in seconds")
    sector_3_time: Optional[float] = Field(None, description="Sector 3 time in seconds")
    tire_compound: Optional[str] = Field(None, description="Current tire compound")
    fuel_load: Optional[float] = Field(None, description="Current fuel load in kg")
    grid_position: Optional[int] = Field(None, description="Starting grid position")
    current_position: Optional[int] = Field(None, description="Current race position")
    air_temperature: Optional[float] = Field(
        None, description="Air temperature in Celsius"
    )
    track_temperature: Optional[float] = Field(
        None, description="Track temperature in Celsius"
    )
    humidity: Optional[float] = Field(None, description="Humidity percentage")
    weather_condition: Optional[str] = Field(None, description="Weather condition")
    track_type: Optional[str] = Field(
        None, description="Track type (street/permanent/temporary)"
    )
    driver_team: Optional[str] = Field(None, description="Driver team name")
    pit_stop_count: int = Field(default=0, description="Number of pit stops completed")
    total_pit_time: float = Field(default=0.0, description="Total time spent in pits")
    lap_status: str = Field(..., description="Lap status (valid/invalid/dnf)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "timestamp": "2024-03-02T15:30:00Z",
                "driver_id": 1,
                "lap_number": 15,
                "lap_time": 78.456,
                "sector_1_time": 25.123,
                "sector_2_time": 26.789,
                "sector_3_time": 26.544,
                "tire_compound": "soft",
                "fuel_load": 45.2,
                "grid_position": 1,
                "current_position": 1,
                "air_temperature": 25.5,
                "track_temperature": 35.2,
                "humidity": 45.0,
                "weather_condition": "dry",
                "track_type": "permanent",
                "driver_team": "Red Bull Racing",
                "pit_stop_count": 1,
                "total_pit_time": 2.8,
                "lap_status": "valid",
            }
        }
    }


class DataProcessingSummary(BaseModel):
    """Schema for data processing summary statistics."""

    session_key: int = Field(..., description="Session identifier")
    total_data_points: int = Field(
        ..., description="Total number of processed data points"
    )
    total_drivers: int = Field(..., description="Total number of drivers")
    total_laps: int = Field(..., description="Total number of laps")
    data_sources: List[str] = Field(
        ..., description="Data sources included in processing"
    )
    processing_time_ms: int = Field(
        ..., description="Time taken for processing in milliseconds"
    )
    missing_data_points: int = Field(..., description="Number of missing data points")
    data_quality_score: float = Field(..., description="Data quality score (0-1)")
    features_generated: List[str] = Field(..., description="List of features generated")
    processing_errors: List[str] = Field(
        default_factory=list, description="List of processing errors"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "session_key": 9472,
                "total_data_points": 1040,
                "total_drivers": 20,
                "total_laps": 52,
                "data_sources": ["weather", "grid", "lap_times", "pit_stops"],
                "processing_time_ms": 1250,
                "missing_data_points": 15,
                "data_quality_score": 0.95,
                "features_generated": [
                    "lap_time_normalized",
                    "fuel_load_normalized",
                    "tire_wear_estimate",
                    "position_change",
                    "weather_impact_score",
                ],
                "processing_errors": [],
            }
        }
    }


class DataProcessingResponse(BaseModel):
    """Response schema for data processing results."""

    session_key: int = Field(..., description="Session identifier")
    session_name: str = Field(..., description="Session name")
    track_name: str = Field(..., description="Track name")
    country: str = Field(..., description="Country")
    year: int = Field(..., description="Season year")
    processing_summary: DataProcessingSummary = Field(
        ..., description="Processing summary"
    )
    processed_data: List[ProcessedDataPoint] = Field(
        ..., description="Processed data points"
    )
    feature_columns: List[str] = Field(..., description="List of feature column names")
    target_columns: List[str] = Field(..., description="List of target column names")
    created_at: datetime = Field(..., description="Processing completion timestamp")

    model_config = {
        "json_schema_extra": {
            "example": {
                "session_key": 9472,
                "session_name": "2024 Bahrain Grand Prix",
                "track_name": "Bahrain International Circuit",
                "country": "Bahrain",
                "year": 2024,
                "processing_summary": {},
                "processed_data": [],
                "feature_columns": [
                    "lap_number",
                    "tire_compound_encoded",
                    "fuel_load_normalized",
                    "air_temperature",
                    "track_temperature",
                    "humidity",
                    "weather_condition_encoded",
                    "grid_position",
                    "pit_stop_count",
                    "total_pit_time",
                ],
                "target_columns": [
                    "lap_time",
                    "sector_1_time",
                    "sector_2_time",
                    "sector_3_time",
                ],
                "created_at": "2024-01-15T10:30:00Z",
            }
        }
    }


class CategoricalEncodingRequest(BaseModel):
    """Request schema for categorical feature encoding."""

    session_key: int = Field(..., description="Session identifier")
    feature_name: str = Field(..., description="Categorical feature name to encode")
    encoding_type: str = Field(
        default="onehot",
        description="Encoding type (onehot/label)",
        pattern="^(onehot|label)$",
    )
    include_validation: bool = Field(
        default=True, description="Include validation of encoding consistency"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "session_key": 9472,
                "feature_name": "tire_compound",
                "encoding_type": "onehot",
                "include_validation": True,
            }
        }
    }


class CategoricalMappingResponse(BaseModel):
    """Response schema for categorical feature mappings."""

    session_key: int = Field(..., description="Session identifier")
    feature_name: str = Field(..., description="Categorical feature name")
    encoding_type: str = Field(..., description="Encoding type used")
    categories: List[str] = Field(..., description="List of unique categories")
    feature_mappings: Dict[str, Any] = Field(
        ..., description="Feature mapping dictionary"
    )
    encoded_feature_names: List[str] = Field(
        ..., description="Names of encoded features"
    )
    encoding_metadata: Dict[str, Any] = Field(..., description="Encoding metadata")
    validation_passed: bool = Field(
        ..., description="Whether encoding validation passed"
    )
    created_at: datetime = Field(..., description="Encoding creation timestamp")

    model_config = {
        "json_schema_extra": {
            "example": {
                "session_key": 9472,
                "feature_name": "tire_compound",
                "encoding_type": "onehot",
                "categories": ["soft", "medium", "hard"],
                "feature_mappings": {
                    "tire_compound_soft": [1, 0, 0],
                    "tire_compound_medium": [0, 1, 0],
                    "tire_compound_hard": [0, 0, 1],
                },
                "encoded_feature_names": [
                    "tire_compound_soft",
                    "tire_compound_medium",
                    "tire_compound_hard",
                ],
                "encoding_metadata": {
                    "feature_count": 3,
                    "sparse_encoding": False,
                    "handle_unknown": "ignore",
                },
                "validation_passed": True,
                "created_at": "2024-01-15T10:30:00Z",
            }
        }
    }


class EncodingValidationResponse(BaseModel):
    """Response schema for encoding validation results."""

    session_key: int = Field(..., description="Session identifier")
    total_features_validated: int = Field(..., description="Total features validated")
    validation_passed: bool = Field(..., description="Overall validation status")
    feature_validations: Dict[str, bool] = Field(
        ..., description="Per-feature validation status"
    )
    encoding_consistency: Dict[str, str] = Field(
        ..., description="Encoding consistency report"
    )
    validation_errors: List[str] = Field(..., description="List of validation errors")
    validation_time_ms: int = Field(..., description="Validation time in milliseconds")

    model_config = {
        "json_schema_extra": {
            "example": {
                "session_key": 9472,
                "total_features_validated": 5,
                "validation_passed": True,
                "feature_validations": {
                    "weather_condition": True,
                    "tire_compound": True,
                    "track_type": True,
                    "driver_team": True,
                    "lap_status": True,
                },
                "encoding_consistency": {
                    "weather_condition": "consistent_onehot_encoding",
                    "tire_compound": "consistent_onehot_encoding",
                    "track_type": "consistent_onehot_encoding",
                    "driver_team": "consistent_onehot_encoding",
                    "lap_status": "consistent_onehot_encoding",
                },
                "validation_errors": [],
                "validation_time_ms": 45,
            }
        }
    }
