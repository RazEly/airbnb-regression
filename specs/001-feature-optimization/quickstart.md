# Quickstart: Feature Optimization

This guide explains how to develop and test the new robustness features using the **Ephemeral Testing Workflow**.

## Prerequisites

- Python 3.12+
- PySpark installed
- `airbnb.csv` in the root directory

## Development Workflow

1.  **Create Ephemeral Test File**:
    Create `testing.py` in the root directory.

    ```python
    # testing.py
    from pyspark.sql import SparkSession
    import pyspark.sql.functions as F
    from data_transformation import load_data, initial_selection # Import existing helpers if needed

    spark = SparkSession.builder.appName("TestFeature").getOrCreate()
    df = load_data("airbnb.csv", sample_fraction=0.1) # Work with sample
    
    # ... Implement your new logic here ...
    # e.g., define function: def filter_outliers(df): ...
    # Apply and print(df.count())
    ```

2.  **Implement Logic**:
    Write your transformation functions (e.g., `impute_structural`, `filter_outliers_iqr`) in `testing.py`.

3.  **Verify**:
    Run the script:
    ```bash
    python testing.py
    ```
    Check output counts and schemas against `data-model.md`.

4.  **Integrate**:
    Once verified, copy the functions to the **Definitions** section of `data_transformation.py` and add them to the pipeline in `apply_stateless_transformations` (or equivalent).

5.  **Cleanup**:
    Delete `testing.py` before committing.

## Key Commands

- Run Production Pipeline: `python data_transformation.py`