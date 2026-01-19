# Phase 0: Research Findings

## 1. Outlier Management in PySpark

**Decision**: Use `approxQuantile` to calculate bounds and `filter` to remove outliers.
**Rationale**: `approxQuantile` is designed for distributed datasets and avoids pulling all data to the driver. It allows for efficient IQR calculation with a tunable `relativeError` (e.g., 0.01) which provides a good balance between accuracy and performance on large data.
**Alternatives Considered**:
- **Z-Score**: Assumes normal distribution, which price data often violates (long tail).
- **Absolute Thresholds**: While `analysis.md` mentioned $500, the spec clarified a preference for a dynamic IQR-based approach for robustness.

## 2. Structural Imputation

**Decision**: Use `fillna(0)` for structural columns (`beds`, `bedrooms`, `bathrooms`).
**Rationale**: This semantically maps "missing" to "none" (e.g., a studio has 0 separate bedrooms). It is natively optimized in Spark and avoids the overhead of the `Imputer` estimator for simple constant filling.
**Decision**: Use `Imputer(strategy="median")` for `host_response_rate`.
**Rationale**: Response rates are heavily skewed (often capped at 100%), so median is more robust than mean.

## 3. Amenities Feature Engineering

**Decision**: Use native Spark higher-order functions (`split`, `regexp_replace`, `array_contains`, `aggregate`) instead of Python UDFs.
**Rationale**: Native functions run directly in the JVM, avoiding the serialization/deserialization overhead of Python UDFs. This leverages the Catalyst optimizer for significant performance gains.
**Implementation Detail**:
- **Parsing**: `split(regexp_replace(col("amenities"), "[\"\\\[\\\\]]", ""), ",")` to convert the stringified list to a true array.
- **Grouping**: Use `array_contains` to check for the presence of keywords (e.g., "Pool", "Wifi") and sum the booleans (casted to integer) to create group scores.

## 4. Host Longevity

**Decision**: Calculate `days_since_hosted` using `datediff(current_date(), to_date(col("host_since")))`.
**Rationale**: This uses standard, efficient Spark SQL functions.
**Handling Missing**: If `host_since` is missing, impute with the median longevity (calculated via `approxQuantile`) to prevent data loss.