"""
Custom exceptions for F1 What-If Simulator API.
"""


class F1SimulatorError(Exception):
    """Base exception for F1 What-If Simulator."""

    def __init__(self, message: str, code: str = "UNKNOWN_ERROR"):
        self.message = message
        self.code = code
        super().__init__(self.message)


class DriverNotFoundError(F1SimulatorError):
    """Raised when a driver is not found."""

    def __init__(self, driver_id: int):
        self.driver_id = driver_id
        super().__init__(f"Driver with ID {driver_id} not found", "DRIVER_NOT_FOUND")


class TrackNotFoundError(F1SimulatorError):
    """Raised when a track is not found."""

    def __init__(self, track_id: int):
        self.track_id = track_id
        super().__init__(f"Track with ID {track_id} not found", "TRACK_NOT_FOUND")


class InvalidSimulationParametersError(F1SimulatorError):
    """Raised when simulation parameters are invalid."""

    def __init__(self, details: str):
        self.details = details
        super().__init__(
            f"Invalid simulation parameters: {details}", "INVALID_SIMULATION_PARAMETERS"
        )


class ModelLoadError(F1SimulatorError):
    """Raised when the ML model fails to load."""

    def __init__(self, model_path: str, error: str):
        self.model_path = model_path
        self.error = error
        super().__init__(
            f"Failed to load model from {model_path}: {error}", "MODEL_LOAD_ERROR"
        )


class OpenF1APIError(F1SimulatorError):
    """Raised when OpenF1 API calls fail."""

    def __init__(self, message: str, status_code: int = 0):
        self.status_code = status_code
        super().__init__(message, "OPENF1_API_ERROR")


class SimulationExecutionError(F1SimulatorError):
    """Raised when simulation execution fails."""

    def __init__(self, simulation_id: str, error: str):
        self.simulation_id = simulation_id
        self.error = error
        super().__init__(
            f"Simulation {simulation_id} failed: {error}", "SIMULATION_EXECUTION_ERROR"
        )


class CacheError(F1SimulatorError):
    """Raised when cache operations fail."""

    def __init__(self, operation: str, error: str):
        self.operation = operation
        self.error = error
        super().__init__(f"Cache {operation} failed: {error}", "CACHE_ERROR")
