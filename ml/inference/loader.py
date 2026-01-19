"""
Model Loader Module

Loads all ML pipeline artifacts and provides inference utilities:
- PySpark models (GBT, Imputers, Scaler)
- VectorAssemblers (reconstructed from configs)
- Lookup dictionaries (city medians, cluster medians)
- KNN classifiers for cluster assignment

Author: ML Integration Team
Date: 2026-01-19
"""

import json
import os
from typing import Optional, Dict, Any, Tuple

import pandas as pd
from pyspark.ml.regression import GBTRegressionModel
from pyspark.ml.feature import ImputerModel, StandardScalerModel, VectorAssembler
from sklearn.neighbors import KNeighborsClassifier
import numpy as np


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate distance between two points on Earth using Haversine formula.

    Args:
        lat1, lon1: First point coordinates (degrees)
        lat2, lon2: Second point coordinates (degrees)

    Returns:
        Distance in kilometers
    """
    from math import radians, sin, cos, sqrt, atan2

    R = 6371  # Earth's radius in km

    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return R * c


class ModelLoader:
    """
    Loads and manages all ML model artifacts for inference.

    Usage:
        loader = ModelLoader(spark, model_dir="./models")
        cluster_id = loader.get_cluster_id("Greater London", 51.5074, -0.1278)
        median = loader.get_city_median("Greater London")
    """

    def __init__(self, spark, model_dir="./models"):
        """
        Initialize the model loader.

        Args:
            spark: SparkSession instance
            model_dir: Path to directory containing model artifacts
        """
        self.spark = spark
        self.model_dir = model_dir

        print(f"Loading ML models from {model_dir}...")

        # Load PySpark models
        print("  [1/7] Loading GBT model...")
        self.gbt_model = GBTRegressionModel.load(f"{model_dir}/gbt_model")

        print("  [2/7] Loading imputers...")
        self.imputer_continuous = ImputerModel.load(f"{model_dir}/imputer_continuous")
        self.imputer_binary = ImputerModel.load(f"{model_dir}/imputer_binary")

        print("  [3/7] Loading scaler...")
        self.scaler = StandardScalerModel.load(f"{model_dir}/scaler_continuous")

        # Load VectorAssembler configurations
        print("  [4/7] Loading assembler configurations...")
        with open(f"{model_dir}/assembler_configs.json", "r") as f:
            assembler_configs = json.load(f)

        # Reconstruct VectorAssemblers from configurations
        self.assembler_continuous = VectorAssembler(
            inputCols=assembler_configs["assembler_continuous"]["input_cols"],
            outputCol=assembler_configs["assembler_continuous"]["output_col"],
        )
        self.assembler_binary = VectorAssembler(
            inputCols=assembler_configs["assembler_binary"]["input_cols"],
            outputCol=assembler_configs["assembler_binary"]["output_col"],
        )
        self.assembler_final = VectorAssembler(
            inputCols=assembler_configs["assembler_final"]["input_cols"],
            outputCol=assembler_configs["assembler_final"]["output_col"],
        )

        # Load lookup dictionaries
        print("  [5/7] Loading lookup dictionaries...")
        with open(f"{model_dir}/city_medians.json", "r") as f:
            self.city_medians = json.load(f)

        with open(f"{model_dir}/global_median.json", "r") as f:
            self.global_median = json.load(f)["global_median"]

        with open(f"{model_dir}/cluster_medians.json", "r") as f:
            self.cluster_medians = json.load(f)

        # Load city center coordinates for distance-based matching
        with open(f"{model_dir}/city_centers.json", "r") as f:
            self.city_centers = json.load(f)

        with open(f"{model_dir}/top_cities.json", "r") as f:
            self.top_cities = json.load(f)

        # Load cluster data from Parquet
        print("  [6/7] Loading cluster data from Parquet...")
        cluster_pdf = pd.read_parquet(f"{model_dir}/cluster_data.parquet")
        print(f"  - Loaded {len(cluster_pdf):,} cluster points")

        # Build KNN classifiers for cluster assignment (one per city)
        print("  [7/7] Building KNN classifiers for cluster assignment...")
        self.knn_models = {}
        for city in cluster_pdf["city"].unique():
            city_data = cluster_pdf[cluster_pdf["city"] == city]

            # Extract coordinates and cluster IDs
            X = city_data[["lat", "long"]].values
            y = city_data["cluster_id"].values

            # Use KNN with k=1 for exact cluster assignment (same as before)
            knn = KNeighborsClassifier(n_neighbors=1, metric="euclidean")
            knn.fit(X, y)
            self.knn_models[city] = knn

        print(f"  ✓ Built KNN models for {len(self.knn_models)} cities")

        # Load metadata
        with open(f"{model_dir}/metadata.json", "r") as f:
            self.metadata = json.load(f)

        print(f"✓ Model loaded successfully!")
        print(f"  - Trained on: {self.metadata['training_date']}")
        print(f"  - Cities: {self.metadata['num_cities']}")
        print(f"  - Clusters: {self.metadata['num_clusters']}")
        print(
            f"  - Performance: R²={self.metadata['performance_metrics']['R2']:.3f}, "
            f"RMSE={self.metadata['performance_metrics']['RMSE']:.3f}"
        )

    def match_city_by_distance(self, lat: float, lon: float) -> Optional[str]:
        """
        Match a listing to the nearest city using Haversine distance.

        Args:
            lat: Listing latitude
            lon: Listing longitude

        Returns:
            Nearest city name, or None if coordinates are invalid

        Examples:
            >>> loader.match_city_by_distance(51.5074, -0.1278)
            "Greater London"
            >>> loader.match_city_by_distance(40.7128, -74.0060)
            "New York"
            >>> loader.match_city_by_distance(None, None)
            None
        """
        if lat is None or lon is None:
            return None

        min_distance = float("inf")
        nearest_city = None

        for city, coords in self.city_centers.items():
            distance = haversine_distance(lat, lon, coords["lat"], coords["lon"])
            if distance < min_distance:
                min_distance = distance
                nearest_city = city

        return nearest_city

    def get_cluster_id(
        self, city: Optional[str], lat: Optional[float], lon: Optional[float]
    ) -> int:
        """
        Get cluster ID for a listing using KNN on (lat, long).

        Args:
            city: Matched city name (from match_city_by_distance)
            lat: Latitude
            lon: Longitude

        Returns:
            int: Cluster ID, or -1 if city not found or coordinates missing

        Examples:
            >>> loader.get_cluster_id("Greater London", 51.5074, -0.1278)
            42
            >>> loader.get_cluster_id("Unknown City", 51.5074, -0.1278)
            -1
            >>> loader.get_cluster_id("Greater London", None, None)
            -1
        """
        # Return -1 if missing data
        if not city or lat is None or lon is None:
            return -1

        # Check if we have a KNN model for this city
        if city not in self.knn_models:
            return -1

        # Predict cluster using KNN
        knn = self.knn_models[city]
        X = np.array([[lat, lon]])
        cluster_id = knn.predict(X)[0]

        return int(cluster_id)

    def get_city_median(self, city: Optional[str]) -> float:
        """
        Get median price for a city (log-scale).

        Args:
            city: Matched city name (from match_city_by_distance) or None

        Returns:
            float: Median price (log-scale), or global median if city is None or not found
        """
        if not city or city not in self.city_medians:
            return self.global_median

        return self.city_medians[city]

    def get_cluster_median(self, city: Optional[str], cluster_id: int) -> float:
        """
        Get median price for a cluster (log-scale).

        Args:
            city: Matched city name
            cluster_id: Cluster ID from get_cluster_id

        Returns:
            float: Cluster median price (log-scale), or global median if not found
        """
        if not city or cluster_id == -1:
            return self.global_median

        key = f"{city}|{cluster_id}"
        return self.cluster_medians.get(key, self.global_median)


if __name__ == "__main__":
    # Test the model loader
    from pyspark.sql import SparkSession

    # Create Spark session
    spark = (
        SparkSession.builder.appName("ModelLoaderTest")
        .master("local[*]")
        .config("spark.driver.memory", "2g")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("ERROR")

    # Load models
    loader = ModelLoader(spark, model_dir="../../models")

    # Test city matching (now using coordinates)
    print("\n--- Testing City Matching (Distance-Based) ---")
    test_locations = [
        ("Greater London", 51.5074, -0.1278),
        ("New York", 40.7128, -74.0060),
        ("Paris", 48.8566, 2.3522),
        ("Unknown", 0.0, 0.0),  # Middle of Atlantic Ocean
    ]
    for name, lat, lon in test_locations:
        match = loader.match_city_by_distance(lat, lon)
        print(f"{name:20s} ({lat:7.4f}, {lon:8.4f}) -> {match}")

    # Test cluster assignment
    print("\n--- Testing Cluster Assignment ---")
    matched_city = loader.match_city_by_distance(51.5074, -0.1278)
    cluster = loader.get_cluster_id(matched_city, 51.5074, -0.1278)
    print(f"City: {matched_city}, Cluster: {cluster}")

    # Test medians
    print("\n--- Testing Median Lookups ---")
    city_median = loader.get_city_median(matched_city)
    cluster_median = loader.get_cluster_median(matched_city, cluster)
    print(f"City median: {city_median:.4f} (log-scale)")
    print(f"Cluster median: {cluster_median:.4f} (log-scale)")

    spark.stop()
