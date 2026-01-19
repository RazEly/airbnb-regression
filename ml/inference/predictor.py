"""
Price Predictor Module

Handles ML price prediction for Airbnb listings using PySpark models.

Architecture:
- Initializes SparkSession once on app startup
- Loads all ML models (GBT, Imputers, Scaler) into memory
- Provides predict() function for inference

Author: ML Integration Team
Date: 2026-01-19
"""

import logging
import os
from typing import Dict, Any, Optional
import numpy as np

from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType,
    StructField,
    DoubleType,
    IntegerType,
)

from ml.inference.loader import ModelLoader
from ml.inference.features import prepare_features_for_model
from ml.utils.currency import convert_to_usd

logger = logging.getLogger(__name__)


class PricePredictor:
    """
    ML Price Predictor using PySpark GBT model.

    Usage:
        predictor = PricePredictor(model_dir="./models")
        result = predictor.predict(parsed_listing_data)
    """

    def __init__(self, model_dir: str):
        """
        Initialize predictor and load all models.

        Args:
            model_dir: Path to directory containing model artifacts

        Raises:
            Exception: If models fail to load
        """
        self.model_dir = model_dir

        # Suppress Py4J debug messages BEFORE initializing Spark
        logging.getLogger("py4j").setLevel(logging.ERROR)
        logging.getLogger("py4j.clientserver").setLevel(logging.ERROR)
        logging.getLogger("py4j.java_gateway").setLevel(logging.ERROR)

        logger.info("=" * 70)
        logger.info("INITIALIZING ML PRICE PREDICTOR")
        logger.info("=" * 70)

        # Step 1: Initialize Spark
        logger.info("\n[1/3] Initializing SparkSession...")
        self.spark = (
            SparkSession.builder.appName("AirbnbPricePredictor")
            .config("spark.driver.memory", "2g")
            .config("spark.sql.shuffle.partitions", "8")
            .config("spark.ui.enabled", "false")
            .getOrCreate()
        )

        # Suppress Spark logging noise
        self.spark.sparkContext.setLogLevel("ERROR")

        logger.info("  ✓ SparkSession ready")

        # Step 2: Load models
        logger.info("\n[2/3] Loading ML models...")
        self.model_loader = ModelLoader(self.spark, model_dir)
        logger.info("  ✓ All models loaded")

        # Step 3: Define schema for inference
        logger.info("\n[3/3] Setting up inference pipeline...")
        self._setup_schema()
        logger.info("  ✓ Inference pipeline ready")

        logger.info("\n" + "=" * 70)
        logger.info("ML PREDICTOR INITIALIZED SUCCESSFULLY")
        logger.info("=" * 70 + "\n")

    def _setup_schema(self):
        """Define Spark DataFrame schema for features."""
        # Schema matches the 19 features (18 continuous + 1 binary)
        self.feature_schema = StructType(
            [
                # Continuous features
                StructField("review_volume_quality", DoubleType(), True),
                StructField("num_bedrooms", DoubleType(), True),
                StructField("median_city", DoubleType(), True),
                StructField("loc_details_length_logp1", DoubleType(), True),
                StructField("guests", DoubleType(), True),
                StructField("amenities_count", DoubleType(), True),
                StructField("description_length_logp1", DoubleType(), True),
                StructField("cluster_median", DoubleType(), True),
                StructField("host_number_of_reviews", DoubleType(), True),
                StructField("ratings", DoubleType(), True),
                StructField("host_rating", DoubleType(), True),
                StructField("host_year", DoubleType(), True),
                StructField("rooms_per_guest", DoubleType(), True),
                StructField("property_number_of_reviews", DoubleType(), True),
                StructField("total_rooms", DoubleType(), True),
                StructField("lat", DoubleType(), True),
                StructField("long", DoubleType(), True),
                StructField("num_baths", DoubleType(), True),
                # Binary feature
                StructField("is_superhost_binary", IntegerType(), True),
            ]
        )

    def validate_required_fields(self, parsed_data: Dict[str, Any]) -> list:
        """
        Check if critical fields are present.

        Args:
            parsed_data: Output from parse_listing_document()

        Returns:
            List of missing critical field names
        """
        critical = ["city", "lat", "long", "price", "currency"]
        data = parsed_data.get("data", {})
        missing = [f for f in critical if data.get(f) is None]
        return missing

    def predict(
        self, parsed_data: Dict[str, Any], verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Run full prediction pipeline.

        Pipeline:
        1. Validate required fields
        2. Engineer features (19 features)
        3. Create Spark DataFrame
        4. Apply imputation (handle None values)
        5. Apply scaling (standardize continuous features)
        6. Run GBT prediction
        7. Inverse log transform (expm1)
        8. Calculate comparison metrics

        Args:
            parsed_data: Output from parse_listing_document()
            verbose: If True, log detailed steps

        Returns:
            Dict with:
                - predicted_price_per_night_usd: Predicted price in USD
                - listed_price_per_night_usd: Listed price in USD
                - difference_usd: Dollar difference
                - difference_pct: Percentage difference
                - features: Dict of all 19 engineered features
                - city: Matched city name
                - cluster_id: Cluster ID
                - error: Error message if prediction failed
        """
        try:
            # Step 1: Validate
            missing = self.validate_required_fields(parsed_data)
            if missing:
                error_msg = f"Missing critical fields: {missing}"
                logger.warning(f"Prediction failed: {error_msg}")
                return {"error": error_msg, "features": {}}

            # Step 2: Engineer features
            if verbose:
                logger.info("Running feature engineering pipeline...")

            features = prepare_features_for_model(
                parsed_data, self.model_loader, verbose=verbose
            )

            # Step 3: Extract listed price for comparison
            data = parsed_data.get("data", {})
            pricing = data.get("pricing_details", {})
            price_per_night = pricing.get("price_per_night")
            currency = data.get("currency", "USD")

            if price_per_night is None:
                return {"error": "No price_per_night found", "features": features}

            listed_price_usd = convert_to_usd(price_per_night, currency)

            # Step 4: Create Spark DataFrame from features
            if verbose:
                logger.info("\nCreating Spark DataFrame for prediction...")

            # Convert features to row format
            row_data = []
            for field in self.feature_schema.fields:
                field_name = field.name
                value = features.get(field_name)
                # Convert int to float for DoubleType fields, keep None as-is
                if isinstance(field.dataType, DoubleType) and value is not None:
                    value = float(value)
                elif isinstance(field.dataType, IntegerType) and value is not None:
                    value = int(value)
                row_data.append(value)

            # Create DataFrame
            df = self.spark.createDataFrame([row_data], schema=self.feature_schema)

            if verbose:
                logger.info(
                    f"  ✓ Created DataFrame with {df.count()} row, {len(df.columns)} columns"
                )

            # Step 5: Apply continuous imputation
            if verbose:
                logger.info("\nApplying imputation...")

            continuous_cols = [
                f.name
                for f in self.feature_schema.fields
                if isinstance(f.dataType, DoubleType)
            ]
            binary_col = "is_superhost_binary"

            # Apply continuous imputer (original cols → cols_imputed)
            df = self.model_loader.imputer_continuous.transform(df)

            # Apply binary imputer (original col → col_imputed)
            df = self.model_loader.imputer_binary.transform(df)

            if verbose:
                logger.info("  ✓ Imputation complete")

            # Step 6: Assemble imputed continuous features into vector
            if verbose:
                logger.info("\nAssembling imputed features into vectors...")

            df = self.model_loader.assembler_continuous.transform(df)

            # Assemble imputed binary feature into vector
            df = self.model_loader.assembler_binary.transform(df)

            if verbose:
                logger.info("  ✓ Feature assembly complete")

            # Step 7: Apply scaling to continuous features
            if verbose:
                logger.info("\nApplying feature scaling...")

            df = self.model_loader.scaler.transform(df)

            if verbose:
                logger.info("  ✓ Scaling complete")

            # Step 8: Assemble final feature vector (scaled_continuous + binary)
            if verbose:
                logger.info("\nAssembling final feature vector...")

            df = self.model_loader.assembler_final.transform(df)

            if verbose:
                logger.info("  ✓ Feature vector ready")

            # Step 9: Run GBT prediction
            if verbose:
                logger.info("\nRunning GBT model prediction...")

            predictions = self.model_loader.gbt_model.transform(df)

            # Extract prediction
            prediction_row = predictions.select("prediction").first()
            log_price = prediction_row[0]

            # Step 9: Inverse log transform
            predicted_price_usd = np.expm1(log_price)

            if verbose:
                logger.info(
                    f"  ✓ Prediction complete: ${predicted_price_usd:.2f} USD/night"
                )

            # Step 10: Calculate comparison metrics
            difference_usd = predicted_price_usd - listed_price_usd
            difference_pct = (
                (difference_usd / listed_price_usd) * 100
                if listed_price_usd and listed_price_usd > 0
                else 0
            )

            # Return results
            return {
                "predicted_price_per_night_usd": predicted_price_usd,
                "listed_price_per_night_usd": listed_price_usd,
                "difference_usd": difference_usd,
                "difference_pct": difference_pct,
                "features": features,
                "city": features.get("city_name", "Unknown"),
                "cluster_id": features.get("cluster_id"),
                "currency": currency,
                "listed_price_original": price_per_night,
            }

        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            import traceback

            traceback.print_exc()
            return {
                "error": str(e),
                "features": features if "features" in locals() else {},
            }

    def __del__(self):
        """Clean up Spark session on shutdown."""
        if hasattr(self, "spark"):
            self.spark.stop()
            logger.info("SparkSession stopped")
