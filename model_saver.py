"""
Model Saver Module

Saves all trained ML pipeline components for deployment:
- PySpark ML models (GBT, Imputers, Scaler)
- VectorAssemblers (serialized)
- Lookup dictionaries (JSON)
- Cluster data for KNN
- Metadata

Author: ML Pipeline Team
Date: 2026-01-19
"""

import os
import json
from datetime import datetime
import pyspark.sql.functions as F


def save_pipeline_artifacts(
    gbt_model,
    imputer_continuous_model,
    imputer_binary_model,
    scaler_model,
    assembler_continuous,
    assembler_binary,
    assembler_final,
    train_df,
    city_medians_dict,
    city_centers_dict,
    global_median,
    continuous_features,
    binary_features,
    performance_metrics,
    output_dir="./models",
):
    """
    Save all pipeline components to disk.

    Creates directory structure:
    models/
    ├── gbt_model/               (PySpark GBTRegressionModel)
    ├── imputer_continuous/      (PySpark ImputerModel)
    ├── imputer_binary/          (PySpark ImputerModel)
    ├── scaler_continuous/       (PySpark StandardScalerModel)
    ├── assembler_configs.json   (VectorAssembler configurations)
    ├── city_medians.json
    ├── city_centers.json        (NEW: City center coordinates)
    ├── global_median.json
    ├── cluster_medians.json
    ├── cluster_data.json
    ├── top_cities.json
    └── metadata.json

    Args:
        gbt_model: Trained GBTRegressionModel
        imputer_continuous_model: ImputerModel for continuous features
        imputer_binary_model: ImputerModel for binary features
        scaler_model: StandardScalerModel
        assembler_continuous: VectorAssembler for continuous features
        assembler_binary: VectorAssembler for binary features
        assembler_final: Final VectorAssembler
        train_df: Training DataFrame (for cluster extraction)
        city_medians_dict: Dict of city -> median price (log-scale)
        city_centers_dict: Dict of city -> {"lat": x, "lon": y} (median coordinates)
        global_median: Global median price (log-scale)
        continuous_features: List of continuous feature names
        binary_features: List of binary feature names
        performance_metrics: Dict with model performance metrics
        output_dir: Directory to save all artifacts
    """

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n{'=' * 70}")
    print(f"SAVING PIPELINE ARTIFACTS TO: {output_dir}")
    print(f"{'=' * 70}\n")

    # 1. Save PySpark models
    print("[1/9] Saving PySpark models...")
    gbt_model.write().overwrite().save(f"{output_dir}/gbt_model")
    print("  ✓ GBT model saved")

    imputer_continuous_model.write().overwrite().save(
        f"{output_dir}/imputer_continuous"
    )
    print("  ✓ Continuous imputer saved")

    imputer_binary_model.write().overwrite().save(f"{output_dir}/imputer_binary")
    print("  ✓ Binary imputer saved")

    scaler_model.write().overwrite().save(f"{output_dir}/scaler_continuous")
    print("  ✓ Scaler saved")

    # 2. Save VectorAssembler configurations (column names only)
    print("\n[2/9] Saving VectorAssembler configurations...")

    # PySpark VectorAssemblers can't be pickled, so save their configurations
    assembler_configs = {
        "assembler_continuous": {
            "input_cols": assembler_continuous.getInputCols(),
            "output_col": assembler_continuous.getOutputCol(),
        },
        "assembler_binary": {
            "input_cols": assembler_binary.getInputCols(),
            "output_col": assembler_binary.getOutputCol(),
        },
        "assembler_final": {
            "input_cols": assembler_final.getInputCols(),
            "output_col": assembler_final.getOutputCol(),
        },
    }

    with open(f"{output_dir}/assembler_configs.json", "w") as f:
        json.dump(assembler_configs, f, indent=2)

    print("  ✓ Assembler configurations saved")

    # 3. Save city medians
    print("\n[3/9] Saving city medians...")
    with open(f"{output_dir}/city_medians.json", "w") as f:
        json.dump(city_medians_dict, f, indent=2)
    print(f"  ✓ Saved {len(city_medians_dict)} city medians")

    # 4. Save city centers (NEW: median coordinates per city)
    print("\n[4/9] Saving city centers...")
    with open(f"{output_dir}/city_centers.json", "w") as f:
        json.dump(city_centers_dict, f, indent=2)
    print(f"  ✓ Saved {len(city_centers_dict)} city center coordinates")

    # 5. Save global median
    print("\n[5/9] Saving global median...")
    with open(f"{output_dir}/global_median.json", "w") as f:
        json.dump({"global_median": float(global_median)}, f, indent=2)
    print(f"  ✓ Global median: {global_median:.4f}")

    # 6. Extract and save cluster data
    print("\n[6/9] Extracting and saving cluster data...")
    cluster_data = extract_cluster_training_data(train_df)
    with open(f"{output_dir}/cluster_data.json", "w") as f:
        json.dump(cluster_data, f, indent=2)
    print(f"  ✓ Saved cluster data for {len(cluster_data)} cities")

    # 7. Extract and save cluster medians
    print("\n[7/9] Extracting and saving cluster medians...")
    cluster_medians = extract_cluster_medians(train_df)
    with open(f"{output_dir}/cluster_medians.json", "w") as f:
        json.dump(cluster_medians, f, indent=2)
    print(f"  ✓ Saved {len(cluster_medians)} cluster medians")

    # 8. Extract top cities
    print("\n[8/9] Extracting top cities...")
    top_cities = list(city_medians_dict.keys())
    with open(f"{output_dir}/top_cities.json", "w") as f:
        json.dump(top_cities, f, indent=2)
    print(f"  ✓ Saved {len(top_cities)} top cities")

    # 9. Save metadata
    print("\n[9/9] Saving metadata...")
    metadata = {
        "training_date": datetime.now().isoformat(),
        "continuous_features": continuous_features,
        "binary_features": binary_features,
        "performance_metrics": performance_metrics,
        "num_cities": len(city_medians_dict),
        "num_clusters": len(cluster_medians),
        "global_median": float(global_median),
    }
    with open(f"{output_dir}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print("  ✓ Metadata saved")

    print(f"\n{'=' * 70}")
    print(f"✓ ALL ARTIFACTS SAVED SUCCESSFULLY")
    print(f"{'=' * 70}\n")


def extract_cluster_training_data(train_df):
    """
    Extract cluster training points for KNN assignment.

    Args:
        train_df: Training DataFrame with cluster_id, city, lat, long columns

    Returns:
        {
            "Greater London": [[51.5074, -0.1278, 42], ...],
            "Paris": [[48.8566, 2.3522, 15], ...],
            ...
        }
    """
    print("  - Collecting cluster points from training data...")

    # Collect cluster points grouped by city
    cluster_rows = (
        train_df.filter(F.col("cluster_id") != -1)
        .select("city", "lat", "long", "cluster_id")
        .collect()
    )

    # Group by city
    cluster_data = {}
    for row in cluster_rows:
        city = row["city"]
        lat = float(row["lat"])
        lon = float(row["long"])
        cluster_id = int(row["cluster_id"])

        if city not in cluster_data:
            cluster_data[city] = []

        cluster_data[city].append([lat, lon, cluster_id])

    print(
        f"  - Extracted {len(cluster_rows)} cluster points from {len(cluster_data)} cities"
    )
    return cluster_data


def extract_cluster_medians(train_df):
    """
    Extract cluster medians from training data.

    Args:
        train_df: Training DataFrame with cluster_id, city, price_cleaned columns

    Returns:
        {
            "Greater London|42": 4.8234,  # log-transformed median
            "Greater London|15": 4.9512,
            ...
        }
    """
    print("  - Computing cluster medians...")

    cluster_median_rows = (
        train_df.filter(F.col("cluster_id") != -1)
        .groupBy("city", "cluster_id")
        .agg(F.percentile_approx("price_cleaned", 0.5).alias("median"))
        .collect()
    )

    cluster_medians = {}
    for row in cluster_median_rows:
        city = row["city"]
        cluster_id = int(row["cluster_id"])
        median = float(row["median"])
        key = f"{city}|{cluster_id}"
        cluster_medians[key] = median

    print(f"  - Computed medians for {len(cluster_medians)} clusters")
    return cluster_medians
