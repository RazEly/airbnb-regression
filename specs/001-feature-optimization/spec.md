# Feature Specification: Feature Optimization & Model Robustness

**Feature Branch**: `001-feature-optimization`  
**Created**: 2026-01-01  
**Status**: Draft  
**Input**: User description: "This projects main idea is to create the most robust regression model for airbnb price prediction. There are already many features implemented, some might be redundant and some are very useful. read the @analysis.md file for suggested features to be added to the @data_transformation.py file. anaylze the suggestions based on a web search, compare them with current implemented features. and create a plan for adding features or removing redundancies if necessary"

## Clarifications

### Session 2026-01-01
- Q: Which specific amenity groups must be implemented? → A: Fixed set: Essentials, Luxury, Security, Family-Friendly, Pet-Friendly.
- Q: Should the implementation enforce a hard $500 limit or calculate it dynamically? → A: Dynamic IQR: Remove price > Q3 + 1.5 * IQR.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Robust Outlier Management (Priority: P1)

As a Data Scientist, I want the pipeline to filter out extreme price outliers so that the regression model is not skewed by non-representative luxury listings or data errors.

**Why this priority**: Extreme values disproportionately affect Root Mean Squared Error (RMSE), making the model less generalizable to typical listings.

**Independent Test**:
1. Create a dummy dataset with varying prices to ensure a clear Q3 and IQR.
2. Run the transformation.
3. Verify that listings identified as outliers by the IQR method are removed.

**Acceptance Scenarios**:

1. **Given** a dataset where a listing's price falls above Q3 + 1.5 * IQR, **When** the cleaning pipeline runs, **Then** that listing is excluded.
2. **Given** a dataset where all listing prices fall within Q3 + 1.5 * IQR, **When** the cleaning pipeline runs, **Then** all listings are retained.

---

### User Story 2 - Logical Imputation for Structural Data (Priority: P2)

As a Data Scientist, I want missing structural values (beds, bedrooms, bathrooms) to be filled with 0 instead of the median/mean, so that "missing" correctly implies "feature not present" (e.g., a studio might have missing bedrooms).

**Why this priority**: Median imputation introduces noise by assigning non-existent rooms to properties, confusing the model's understanding of space.

**Independent Test**:
1. Create a dataframe with `null` values in `num_bedrooms`.
2. Run the transformation.
3. Verify `null` values are replaced by `0`.

**Acceptance Scenarios**:

1. **Given** a listing with `num_bedrooms = null`, **When** processed, **Then** `num_bedrooms` becomes `0`.

---

### User Story 3 - Richer Feature Engineering (Priority: P2)

As a Data Scientist, I want to extract grouped amenities (e.g., "Pet Friendly", "Security") and host longevity metrics, so that the model can learn from specific value-adding attributes rather than just a raw count.

**Why this priority**: Raw counts (current implementation) lose information. A "Pool" is worth more than a "Hanger". Grouping captures this semantic value.

**Independent Test**:
1. Create a listing with `amenities = ["Wifi", "Pool", "Dog Bowl"]`.
2. Run the transformation.
3. Verify new columns like `amenity_group_pet_friendly` or `amenity_group_luxury` are created with correct values.

**Acceptance Scenarios**:

1. **Given** a listing with security amenities (Lockbox, Smart Lock), **When** processed, **Then** a `amenities_security_score` (or binary) feature is generated.
2. **Given** a host with `host_since` date, **When** processed, **Then** `days_since_hosted` is calculated.

### Edge Cases

- **Empty Amenities**: If a listing has no amenities (`[]` or null), the system handles it gracefully by assigning 0 scores to all amenity groups.
- **Missing Host Dates**: If `host_since` is missing, the system falls back to `host_year` if available, or imputes the median longevity of the dataset.
- **Extreme/Invalid Prices**: Prices $\le$ $10 (potentially errors or non-stay fees) are filtered out, and upper outliers are removed based on the dynamic IQR calculation.
- **All-Null Structure**: If a listing has no structural info (beds, baths, bedrooms all null), they are all set to 0, treating it as a raw space or data gap.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST filter out listings with Price > Q3 + 1.5 * IQR (Interquartile Range) based on the training data distribution.
- **FR-002**: System MUST impute missing `num_beds`, `num_bedrooms`, and `num_baths` with `0.0` instead of statistical averages.
- **FR-003**: System MUST impute missing `host_response_rate` with the population median.
- **FR-004**: System MUST parse the `amenities` list to create categorical features/scores for the following key groups: Essentials, Luxury, Security, Family-Friendly, Pet-Friendly.
- **FR-005**: System MUST calculate `host_longevity` (e.g., days active) derived from `host_since` or `host_year`.
- **FR-006**: System MUST remove redundant raw columns after feature extraction to keep the dataset clean.

### Key Entities

- **Listing**: Core entity containing price, location, and structural details.
- **Amenities**: Currently a raw string/list; will be transformed into semantic groups.
- **Host**: Source of longevity and response rate features.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Model RMSE on the validation set improves by at least 2% compared to the current baseline.
- **SC-002**: The number of missing values in structural columns (`num_bedrooms`, etc.) is exactly 0 after transformation.
- **SC-003**: Feature importance analysis shows at least one new feature (Amenity Group or Host Longevity) in the top 15 predictors.
- **SC-004**: Pipeline execution time does not increase by more than 20% despite added features.