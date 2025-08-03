"""
Feature engineering service for F1 What-If Simulator.

This module provides comprehensive feature engineering capabilities including:
- Missing value handling and imputation
- Feature creation and transformation
- Data validation and quality checks
- ML-ready feature preparation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import structlog
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_regression

from app.api.v1.schemas import ProcessedDataPoint
from app.core.exceptions import FeatureEngineeringError

logger = structlog.get_logger()


class FeatureEngineeringService:
    """Service class for handling feature engineering operations."""

    def __init__(self):
        """Initialize the feature engineering service."""
        self.imputers: Dict[str, SimpleImputer] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.onehot_encoders: Dict[str, OneHotEncoder] = {}
        self.feature_selectors: Dict[str, SelectKBest] = {}
        self.feature_columns: List[str] = []
        self.target_columns: List[str] = []
        self.categorical_columns: List[str] = []
        self.numerical_columns: List[str] = []
        self.onehot_columns: List[str] = []  # Columns to use one-hot encoding
        self.label_columns: List[str] = []  # Columns to use label encoding
        self._is_fitted = False

    def fit_transform_features(
        self, data_points: List[ProcessedDataPoint], target_column: str = "lap_time"
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Fit the feature engineering pipeline and transform the data.

        Args:
            data_points: List of processed data points
            target_column: Name of the target column for prediction

        Returns:
            Tuple of (features, targets, metadata)
        """
        logger.info(
            "Starting feature engineering pipeline", data_points_count=len(data_points)
        )

        # Convert to DataFrame for easier processing
        df = self._convert_to_dataframe(data_points)

        # Define feature and target columns
        self._define_columns(df, target_column)

        # Handle missing values
        df_imputed = self._handle_missing_values(df)

        # Create engineered features
        df_engineered = self._create_engineered_features(df_imputed)

        # Encode categorical features
        df_encoded = self._encode_categorical_features(df_engineered)

        # Scale numerical features
        df_scaled = self._scale_numerical_features(df_encoded)

        # Select best features
        features, targets = self._select_features(df_scaled, target_column)

        # Prepare metadata
        metadata = self._prepare_metadata(df_scaled, target_column)

        self._is_fitted = True

        logger.info(
            "Feature engineering pipeline completed",
            features_shape=features.shape,
            targets_shape=targets.shape,
        )

        return features, targets, metadata

    def transform_features(
        self, data_points: List[ProcessedDataPoint]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Transform new data using the fitted pipeline.

        Args:
            data_points: List of processed data points

        Returns:
            Tuple of (features, metadata)
        """
        if not self._is_fitted:
            raise FeatureEngineeringError(
                "Feature engineering pipeline must be fitted before transforming"
            )

        logger.info("Transforming features", data_points_count=len(data_points))

        # Convert to DataFrame
        df = self._convert_to_dataframe(data_points)

        # Handle missing values using fitted imputers
        df_imputed = self._handle_missing_values_transform(df)

        # Create engineered features
        df_engineered = self._create_engineered_features_transform(df_imputed)

        # Encode categorical features using fitted encoders
        df_encoded = self._encode_categorical_features_transform(df_engineered)

        # Scale numerical features using fitted scalers
        df_scaled = self._scale_numerical_features_transform(df_encoded)

        # Select features using fitted selectors
        features = self._select_features_transform(df_scaled)

        # Prepare metadata
        metadata = self._prepare_metadata(df_scaled, None)

        logger.info("Feature transformation completed", features_shape=features.shape)

        return features, metadata

    def _convert_to_dataframe(
        self, data_points: List[ProcessedDataPoint]
    ) -> pd.DataFrame:
        """Convert list of ProcessedDataPoint to pandas DataFrame."""
        if not data_points:
            raise FeatureEngineeringError("No data points provided")

        # Convert to list of dictionaries
        data_dicts = [point.model_dump() for point in data_points]
        df = pd.DataFrame(data_dicts)

        logger.info(
            "Converted data to DataFrame",
            shape=df.shape,
            columns=list(df.columns),
        )

        return df

    def _define_columns(self, df: pd.DataFrame, target_column: str):
        """Define feature, target, categorical, and numerical columns."""
        # Target columns
        self.target_columns = [target_column] if target_column in df.columns else []

        # Categorical columns (string/object types)
        self.categorical_columns = [
            col
            for col in df.columns
            if df[col].dtype == "object" or df[col].dtype.name == "category"
        ]

        # Define which categorical columns should use one-hot encoding vs label encoding
        # One-hot encoding for high-cardinality categorical features
        self.onehot_columns = [
            col
            for col in self.categorical_columns
            if col in ["weather_condition", "tire_compound", "lap_status"]
        ]

        # Label encoding for low-cardinality or ordinal categorical features
        self.label_columns = [
            col for col in self.categorical_columns if col not in self.onehot_columns
        ]

        # Numerical columns (exclude target and categorical)
        self.numerical_columns = [
            col
            for col in df.columns
            if col not in self.target_columns + self.categorical_columns
            and pd.api.types.is_numeric_dtype(df[col])
        ]

        # Feature columns (all except target and non-numeric columns like timestamp)
        exclude_columns = self.target_columns + ["timestamp"]
        self.feature_columns = [
            col
            for col in df.columns
            if col not in exclude_columns
            and (col in self.numerical_columns or col in self.categorical_columns)
        ]

        logger.info(
            "Defined column types",
            target_columns=self.target_columns,
            categorical_columns=self.categorical_columns,
            onehot_columns=self.onehot_columns,
            label_columns=self.label_columns,
            numerical_columns=self.numerical_columns,
            feature_columns_count=len(self.feature_columns),
        )

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values using appropriate imputation strategies."""
        df_imputed = df.copy()

        # Handle numerical columns
        for col in self.numerical_columns:
            if col in df_imputed.columns and df_imputed[col].isnull().any():
                if col in [
                    "lap_time",
                    "sector_1_time",
                    "sector_2_time",
                    "sector_3_time",
                ]:
                    # Use median for lap times (more robust to outliers)
                    imputer = SimpleImputer(strategy="median")
                else:
                    # Use mean for other numerical features
                    imputer = SimpleImputer(strategy="mean")

                # Reshape to 2D array for sklearn
                values = df_imputed[col].to_numpy().reshape(-1, 1)
                df_imputed[col] = imputer.fit_transform(values).flatten()
                self.imputers[col] = imputer

                logger.info(
                    f"Imputed missing values in {col}",
                    strategy=imputer.strategy,
                    missing_count=df[col].isnull().sum(),
                )

        # Handle categorical columns
        for col in self.categorical_columns:
            if col in df_imputed.columns and df_imputed[col].isnull().any():
                # Use most frequent value for categorical features
                imputer = SimpleImputer(strategy="most_frequent")
                # Reshape to 2D array for sklearn
                values = df_imputed[col].to_numpy().reshape(-1, 1)
                imputed_values = imputer.fit_transform(values).flatten()
                # Convert back to string type if needed
                if col == "tire_compound" or col == "weather_condition":
                    imputed_values = imputed_values.astype(str)
                df_imputed[col] = imputed_values
                self.imputers[col] = imputer

                logger.info(
                    f"Imputed missing values in {col}",
                    strategy=imputer.strategy,
                    missing_count=df[col].isnull().sum(),
                    before_imputation=df_imputed[col].isnull().sum(),
                    after_imputation=df_imputed[col].isnull().sum(),
                )

        # Handle target column if it's not in feature columns but has missing values
        target_cols = ["lap_time"]  # Add other target columns as needed
        for col in target_cols:
            if (
                col in df_imputed.columns
                and col not in self.numerical_columns
                and df_imputed[col].isnull().any()
            ):
                # Use median for lap times
                imputer = SimpleImputer(strategy="median")
                values = df_imputed[col].to_numpy().reshape(-1, 1)
                df_imputed[col] = imputer.fit_transform(values).flatten()
                self.imputers[col] = imputer

                logger.info(
                    f"Imputed missing values in target column {col}",
                    strategy=imputer.strategy,
                    missing_count=df[col].isnull().sum(),
                )

        return df_imputed

    def _handle_missing_values_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values using fitted imputers."""
        df_imputed = df.copy()

        for col, imputer in self.imputers.items():
            if col in df_imputed.columns and df_imputed[col].isnull().any():
                # Reshape to 2D array for sklearn
                values = df_imputed[col].to_numpy().reshape(-1, 1)
                df_imputed[col] = imputer.transform(values).flatten()

        return df_imputed

    def _create_engineered_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create new engineered features from existing data."""
        df_engineered = df.copy()

        # Time-based features
        if "timestamp" in df_engineered.columns:
            df_engineered["hour_of_day"] = df_engineered["timestamp"].dt.hour
            df_engineered["day_of_week"] = df_engineered["timestamp"].dt.dayofweek
            df_engineered["is_weekend"] = (
                df_engineered["day_of_week"].isin([5, 6]).astype(int)
            )

        # Lap-based features
        if "lap_number" in df_engineered.columns:
            df_engineered["lap_progress"] = (
                df_engineered["lap_number"] / df_engineered["lap_number"].max()
            )
            df_engineered["is_early_lap"] = (df_engineered["lap_number"] <= 5).astype(
                int
            )
            df_engineered["is_late_lap"] = (
                df_engineered["lap_number"] >= df_engineered["lap_number"].max() - 5
            ).astype(int)

        # Performance features
        if all(
            col in df_engineered.columns
            for col in ["sector_1_time", "sector_2_time", "sector_3_time"]
        ):
            df_engineered["total_sector_time"] = (
                df_engineered["sector_1_time"]
                + df_engineered["sector_2_time"]
                + df_engineered["sector_3_time"]
            )
            df_engineered["sector_consistency"] = df_engineered[
                ["sector_1_time", "sector_2_time", "sector_3_time"]
            ].std(axis=1)

        # Position-based features
        if "grid_position" in df_engineered.columns:
            df_engineered["grid_position_normalized"] = (
                df_engineered["grid_position"] / df_engineered["grid_position"].max()
            )
            df_engineered["is_top_10_start"] = (
                df_engineered["grid_position"] <= 10
            ).astype(int)

        # Weather-based features
        if (
            "air_temperature" in df_engineered.columns
            and "track_temperature" in df_engineered.columns
        ):
            df_engineered["temperature_difference"] = (
                df_engineered["track_temperature"] - df_engineered["air_temperature"]
            )

        if "humidity" in df_engineered.columns:
            df_engineered["is_high_humidity"] = (df_engineered["humidity"] > 70).astype(
                int
            )

        # Pit stop features
        if "pit_stop_count" in df_engineered.columns:
            df_engineered["avg_pit_time"] = df_engineered[
                "total_pit_time"
            ] / df_engineered["pit_stop_count"].replace(0, 1)
            df_engineered["has_pit_stopped"] = (
                df_engineered["pit_stop_count"] > 0
            ).astype(int)

        # Tire compound features
        if "tire_compound" in df_engineered.columns:
            df_engineered["tire_age"] = df_engineered.groupby(
                ["driver_id", "tire_compound"]
            ).cumcount()

        # Driver-specific features
        if "driver_id" in df_engineered.columns:
            # Calculate driver's average lap time
            driver_avg_lap = df_engineered.groupby("driver_id")["lap_time"].transform(
                "mean"
            )
            df_engineered["driver_avg_lap_time"] = driver_avg_lap
            df_engineered["lap_time_vs_driver_avg"] = (
                df_engineered["lap_time"] - driver_avg_lap
            )

        # Update feature columns to include new engineered features
        new_features = [col for col in df_engineered.columns if col not in df.columns]
        self.feature_columns.extend(new_features)

        # Update categorical and numerical columns
        self.categorical_columns = (
            df_engineered[self.feature_columns]
            .select_dtypes(include=["object", "category"])
            .columns.tolist()
        )

        self.numerical_columns = (
            df_engineered[self.feature_columns]
            .select_dtypes(include=["int64", "float64"])
            .columns.tolist()
        )

        logger.info(
            "Created engineered features",
            new_features_count=len(new_features),
            total_features=len(self.feature_columns),
        )

        return df_engineered

    def _create_engineered_features_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create engineered features without updating feature_columns (for transform)."""
        df_engineered = df.copy()

        # Time-based features
        if "timestamp" in df_engineered.columns:
            df_engineered["hour_of_day"] = df_engineered["timestamp"].dt.hour
            df_engineered["day_of_week"] = df_engineered["timestamp"].dt.dayofweek
            df_engineered["is_weekend"] = (
                df_engineered["day_of_week"].isin([5, 6]).astype(int)
            )

        # Lap-based features
        if "lap_number" in df_engineered.columns:
            df_engineered["lap_progress"] = (
                df_engineered["lap_number"] / df_engineered["lap_number"].max()
            )
            df_engineered["is_early_lap"] = (df_engineered["lap_number"] <= 5).astype(
                int
            )
            df_engineered["is_late_lap"] = (
                df_engineered["lap_number"] >= df_engineered["lap_number"].max() - 5
            ).astype(int)

        # Performance features
        if all(
            col in df_engineered.columns
            for col in ["sector_1_time", "sector_2_time", "sector_3_time"]
        ):
            df_engineered["total_sector_time"] = (
                df_engineered["sector_1_time"]
                + df_engineered["sector_2_time"]
                + df_engineered["sector_3_time"]
            )
            df_engineered["sector_consistency"] = df_engineered[
                ["sector_1_time", "sector_2_time", "sector_3_time"]
            ].std(axis=1)

        # Position-based features
        if "grid_position" in df_engineered.columns:
            df_engineered["grid_position_normalized"] = (
                df_engineered["grid_position"] / df_engineered["grid_position"].max()
            )
            df_engineered["is_top_10_start"] = (
                df_engineered["grid_position"] <= 10
            ).astype(int)

        # Weather-based features
        if (
            "air_temperature" in df_engineered.columns
            and "track_temperature" in df_engineered.columns
        ):
            df_engineered["temperature_difference"] = (
                df_engineered["track_temperature"] - df_engineered["air_temperature"]
            )

        if "humidity" in df_engineered.columns:
            df_engineered["is_high_humidity"] = (df_engineered["humidity"] > 70).astype(
                int
            )

        # Pit stop features
        if "pit_stop_count" in df_engineered.columns:
            df_engineered["avg_pit_time"] = df_engineered[
                "total_pit_time"
            ] / df_engineered["pit_stop_count"].replace(0, 1)
            df_engineered["has_pit_stopped"] = (
                df_engineered["pit_stop_count"] > 0
            ).astype(int)

        # Tire compound features
        if "tire_compound" in df_engineered.columns:
            df_engineered["tire_age"] = df_engineered.groupby(
                ["driver_id", "tire_compound"]
            ).cumcount()

        # Driver-specific features
        if "driver_id" in df_engineered.columns:
            # Calculate driver's average lap time
            driver_avg_lap = df_engineered.groupby("driver_id")["lap_time"].transform(
                "mean"
            )
            df_engineered["driver_avg_lap_time"] = driver_avg_lap
            df_engineered["lap_time_vs_driver_avg"] = (
                df_engineered["lap_time"] - driver_avg_lap
            )

        return df_engineered

    def _encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features using label encoding and one-hot encoding."""
        df_encoded = df.copy()
        new_feature_columns = []

        # Label encoding for low-cardinality categorical features
        for col in self.label_columns:
            if col in df_encoded.columns:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                self.label_encoders[col] = le
                new_feature_columns.append(col)

                logger.info(
                    f"Label encoded categorical feature {col}",
                    unique_values=len(le.classes_),
                )

        # One-hot encoding for high-cardinality categorical features
        for col in self.onehot_columns:
            if col in df_encoded.columns:
                ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
                # Ensure all values are strings for OneHotEncoder
                df_encoded[col] = df_encoded[col].astype(str)

                # Fit the encoder
                ohe.fit(df_encoded[col].values.reshape(-1, 1))
                self.onehot_encoders[col] = ohe

                # Transform and create new columns
                encoded_values = ohe.transform(df_encoded[col].values.reshape(-1, 1))
                feature_names = [f"{col}_{cat}" for cat in ohe.categories_[0]]

                # Add encoded columns to DataFrame
                for i, feature_name in enumerate(feature_names):
                    df_encoded[feature_name] = encoded_values[:, i]
                    new_feature_columns.append(feature_name)

                # Remove original column
                df_encoded = df_encoded.drop(columns=[col])

                logger.info(
                    f"One-hot encoded categorical feature {col}",
                    unique_values=len(ohe.categories_[0]),
                    new_columns=feature_names,
                )

        # Update feature columns to include new one-hot encoded columns
        # Remove original categorical columns and add new encoded columns
        self.feature_columns = [
            col for col in self.feature_columns if col not in self.categorical_columns
        ] + new_feature_columns

        return df_encoded

    def _encode_categorical_features_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features using fitted encoders."""
        df_encoded = df.copy()

        # Label encoding using fitted encoders
        for col, le in self.label_encoders.items():
            if col in df_encoded.columns:
                # Handle unseen categories by using a default value
                df_encoded[col] = df_encoded[col].astype(str)
                # Use apply instead of map for better handling of missing values
                df_encoded[col] = df_encoded[col].apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else 0
                )

        # One-hot encoding using fitted encoders
        for col, ohe in self.onehot_encoders.items():
            if col in df_encoded.columns:
                # Ensure all values are strings for OneHotEncoder
                df_encoded[col] = df_encoded[col].astype(str)

                # Transform and create new columns
                encoded_values = ohe.transform(df_encoded[col].values.reshape(-1, 1))
                feature_names = [f"{col}_{cat}" for cat in ohe.categories_[0]]

                # Add encoded columns to DataFrame
                for i, feature_name in enumerate(feature_names):
                    df_encoded[feature_name] = encoded_values[:, i]

                # Remove original column
                df_encoded = df_encoded.drop(columns=[col])

        return df_encoded

    def _scale_numerical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scale numerical features using standardization."""
        df_scaled = df.copy()

        for col in self.numerical_columns:
            if col in df_scaled.columns:
                scaler = StandardScaler()
                # Reshape to 2D array for sklearn
                values = df_scaled[col].to_numpy().reshape(-1, 1)
                df_scaled[col] = scaler.fit_transform(values).flatten()
                self.scalers[col] = scaler

                logger.info(f"Scaled numerical feature {col}")

        return df_scaled

    def _scale_numerical_features_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scale numerical features using fitted scalers."""
        df_scaled = df.copy()

        for col, scaler in self.scalers.items():
            if col in df_scaled.columns:
                # Reshape to 2D array for sklearn
                values = df_scaled[col].to_numpy().reshape(-1, 1)
                df_scaled[col] = scaler.transform(values).flatten()

        return df_scaled

    def _select_features(
        self, df: pd.DataFrame, target_column: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Select the best features using statistical tests."""
        if target_column not in df.columns:
            raise FeatureEngineeringError(
                f"Target column {target_column} not found in data"
            )

        # Prepare feature matrix and target vector
        X = df[self.feature_columns].to_numpy().astype(np.float64)
        y = df[target_column].to_numpy().astype(np.float64)

        # Remove rows with NaN values
        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X = X[mask]
        y = y[mask]

        # Select top features (use all features if less than 10)
        k = min(10, len(self.feature_columns))
        selector = SelectKBest(score_func=f_regression, k=k)
        X_selected = selector.fit_transform(X, y)
        self.feature_selectors[target_column] = selector

        # Get selected feature names
        selected_features = [
            self.feature_columns[i] for i in selector.get_support(indices=True)
        ]
        logger.info(
            "Selected best features",
            selected_count=len(selected_features),
            total_features=len(self.feature_columns),
        )

        return X_selected, y

    def _select_features_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Select features using fitted selectors."""
        X = df[self.feature_columns].to_numpy().astype(np.float64)

        # Remove rows with NaN values
        mask = ~np.isnan(X).any(axis=1)
        X = X[mask]

        if self.feature_selectors:
            selector = list(self.feature_selectors.values())[0]
            X_selected: np.ndarray = selector.transform(X)
            return X_selected

        result: np.ndarray = X.astype(np.float64)
        return result

    def _prepare_metadata(
        self, df: pd.DataFrame, target_column: Optional[str]
    ) -> Dict[str, Any]:
        """Prepare metadata about the feature engineering process."""
        metadata = {
            "feature_columns": self.feature_columns,
            "target_columns": self.target_columns,
            "categorical_columns": self.categorical_columns,
            "numerical_columns": self.numerical_columns,
            "total_features": len(self.feature_columns),
            "total_samples": len(df),
            "missing_values_summary": {},
            "feature_importance_scores": {},
            "is_fitted": self._is_fitted,
        }

        # Add missing values summary
        for col in self.feature_columns:
            if col in df.columns:
                missing_count = df[col].isnull().sum()
                missing_percentage = (missing_count / len(df)) * 100
                metadata["missing_values_summary"][col] = {
                    "missing_count": int(missing_count),
                    "missing_percentage": float(missing_percentage),
                }

        # Add feature importance scores if available
        if target_column and target_column in self.feature_selectors:
            selector = self.feature_selectors[target_column]
            scores = selector.scores_
            feature_scores = dict(zip(self.feature_columns, scores))
            metadata["feature_importance_scores"] = feature_scores

        return metadata

    def get_feature_importance(self, target_column: str) -> Dict[str, float]:
        """Get feature importance scores for a target column."""
        if target_column not in self.feature_selectors:
            raise FeatureEngineeringError(
                f"No feature selector found for {target_column}"
            )

        selector = self.feature_selectors[target_column]
        scores = selector.scores_
        feature_scores = dict(zip(self.feature_columns, scores))

        return feature_scores

    def get_data_quality_report(
        self, data_points: List[ProcessedDataPoint]
    ) -> Dict[str, Any]:
        """Generate a comprehensive data quality report."""
        df = self._convert_to_dataframe(data_points)

        report: Dict[str, Any] = {
            "total_records": len(df),
            "total_features": len(df.columns),
            "missing_data_summary": {},
            "data_types": {},
            "value_ranges": {},
            "unique_values": {},
            "data_quality_score": 0.0,
        }

        # Analyze each column
        for col in df.columns:
            col_data = df[col]

            # Missing data
            missing_count = col_data.isnull().sum()
            missing_percentage = (missing_count / len(df)) * 100
            report["missing_data_summary"][col] = {
                "missing_count": int(missing_count),
                "missing_percentage": float(missing_percentage),
            }

            # Data types
            report["data_types"][col] = str(col_data.dtype)

            # Value ranges for numerical data
            if pd.api.types.is_numeric_dtype(col_data):
                report["value_ranges"][col] = {
                    "min": (
                        float(col_data.min()) if not col_data.isnull().all() else None
                    ),
                    "max": (
                        float(col_data.max()) if not col_data.isnull().all() else None
                    ),
                    "mean": (
                        float(col_data.mean()) if not col_data.isnull().all() else None
                    ),
                    "std": (
                        float(col_data.std()) if not col_data.isnull().all() else None
                    ),
                }

            # Unique values for categorical data
            if pd.api.types.is_object_dtype(col_data) or isinstance(
                col_data.dtype, pd.CategoricalDtype
            ):
                unique_vals = col_data.nunique()
                report["unique_values"][col] = {
                    "unique_count": int(unique_vals),
                    "most_common": (
                        col_data.mode().iloc[0] if not col_data.isnull().all() else None
                    ),
                }

        # Calculate overall data quality score
        total_missing = sum(
            report["missing_data_summary"][col]["missing_count"]
            for col in report["missing_data_summary"]
        )
        total_cells = len(df) * len(df.columns)
        report["data_quality_score"] = 1.0 - (total_missing / total_cells)

        return report

    def reset(self):
        """Reset the fitted state and clear all fitted objects."""
        self.imputers.clear()
        self.scalers.clear()
        self.label_encoders.clear()
        self.onehot_encoders.clear()
        self.feature_selectors.clear()
        self.feature_columns.clear()
        self.target_columns.clear()
        self.categorical_columns.clear()
        self.numerical_columns.clear()
        self.onehot_columns.clear()
        self.label_columns.clear()
        self._is_fitted = False

        logger.info("Reset feature engineering service")
