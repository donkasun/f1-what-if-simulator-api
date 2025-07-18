"""
Pydantic schemas for API request/response validation.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, validator


class DriverResponse(BaseModel):
    """Response schema for driver information."""
    
    driver_id: int = Field(..., description="Unique driver identifier")
    name: str = Field(..., description="Driver's full name")
    code: str = Field(..., description="Driver's 3-letter code")
    team: str = Field(..., description="Driver's team name")
    nationality: str = Field(..., description="Driver's nationality")
    
    class Config:
        schema_extra = {
            "example": {
                "driver_id": 1,
                "name": "Max Verstappen",
                "code": "VER",
                "team": "Red Bull Racing",
                "nationality": "Dutch"
            }
        }


class TrackResponse(BaseModel):
    """Response schema for track information."""
    
    track_id: int = Field(..., description="Unique track identifier")
    name: str = Field(..., description="Track name")
    country: str = Field(..., description="Country where the track is located")
    circuit_length: float = Field(..., description="Track length in kilometers")
    number_of_laps: int = Field(..., description="Number of laps in the race")
    
    class Config:
        schema_extra = {
            "example": {
                "track_id": 1,
                "name": "Silverstone Circuit",
                "country": "United Kingdom",
                "circuit_length": 5.891,
                "number_of_laps": 52
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
        regex="^(dry|wet|intermediate)$"
    )
    car_setup: Optional[dict] = Field(
        default_factory=dict,
        description="Car setup parameters (optional)"
    )
    
    @validator('season')
    def validate_season(cls, v):
        """Validate that the season is reasonable."""
        if v < 1950 or v > 2030:
            raise ValueError('Season must be between 1950 and 2030')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "driver_id": 1,
                "track_id": 1,
                "season": 2024,
                "weather_conditions": "dry",
                "car_setup": {
                    "downforce": "high",
                    "tire_compound": "soft"
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
    processing_time_ms: int = Field(..., description="Time taken to process simulation in milliseconds")
    
    class Config:
        schema_extra = {
            "example": {
                "simulation_id": "sim_123456789",
                "driver_id": 1,
                "track_id": 1,
                "season": 2024,
                "predicted_lap_time": 78.456,
                "confidence_score": 0.85,
                "weather_conditions": "dry",
                "car_setup": {
                    "downforce": "high",
                    "tire_compound": "soft"
                },
                "created_at": "2024-01-15T10:30:00Z",
                "processing_time_ms": 1250
            }
        }


class ErrorResponse(BaseModel):
    """Standard error response schema."""
    
    detail: str = Field(..., description="Human-readable error message")
    code: str = Field(..., description="Error code for programmatic handling")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "detail": "Driver with ID 999 not found",
                "code": "DRIVER_NOT_FOUND",
                "timestamp": "2024-01-15T10:30:00Z"
            }
        } 