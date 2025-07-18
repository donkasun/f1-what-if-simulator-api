"""
ML model loader for F1 What-If Simulator.
"""

import os
from typing import Any

import joblib
import structlog

from app.core.config import settings
from app.core.exceptions import ModelLoadError

logger = structlog.get_logger()


class ModelLoader:
    """Handles loading and caching of ML models."""
    
    def __init__(self):
        """Initialize the model loader."""
        self._model = None
        self._model_path = settings.model_path
    
    async def get_model(self) -> Any:
        """
        Get the loaded ML model, loading it if necessary.
        
        Returns:
            Loaded ML model
            
        Raises:
            ModelLoadError: If model loading fails
        """
        if self._model is None:
            await self._load_model()
        
        return self._model
    
    async def _load_model(self) -> None:
        """
        Load the ML model from disk.
        
        Raises:
            ModelLoadError: If model loading fails
        """
        logger.info("Loading ML model", model_path=self._model_path)
        
        try:
            if not os.path.exists(self._model_path):
                # For development, create a dummy model
                logger.warning("Model file not found, creating dummy model", model_path=self._model_path)
                self._model = self._create_dummy_model()
            else:
                self._model = joblib.load(self._model_path)
            
            logger.info("ML model loaded successfully", model_path=self._model_path)
            
        except Exception as e:
            logger.error("Failed to load ML model", model_path=self._model_path, error=str(e), exc_info=True)
            raise ModelLoadError(self._model_path, str(e))
    
    def _create_dummy_model(self) -> Any:
        """
        Create a dummy model for development/testing purposes.
        
        Returns:
            Dummy model that returns a fixed prediction
        """
        from sklearn.ensemble import RandomForestRegressor
        import numpy as np
        
        # Create a simple dummy model
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        
        # Train on dummy data
        X_dummy = np.random.rand(100, 8)  # 8 features
        y_dummy = np.random.uniform(70, 90, 100)  # Lap times between 70-90 seconds
        
        model.fit(X_dummy, y_dummy)
        
        logger.info("Created dummy ML model for development")
        return model
    
    def reload_model(self) -> None:
        """Force reload the model from disk."""
        logger.info("Reloading ML model", model_path=self._model_path)
        self._model = None
        # Note: This is synchronous, but the async get_model will handle the loading
    
    def get_model_info(self) -> dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary containing model information
        """
        if self._model is None:
            return {"status": "not_loaded"}
        
        model_info = {
            "status": "loaded",
            "model_path": self._model_path,
            "model_type": type(self._model).__name__,
        }
        
        # Add model-specific information if available
        if hasattr(self._model, 'n_estimators'):
            model_info["n_estimators"] = str(self._model.n_estimators)
        
        if hasattr(self._model, 'feature_importances_'):
            model_info["feature_importance_count"] = str(len(self._model.feature_importances_))
        
        return model_info 