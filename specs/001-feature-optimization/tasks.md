---
description: "Task list for Feature Optimization & Model Robustness"
---

# Tasks: Feature Optimization & Model Robustness

**Input**: Design documents from `/specs/001-feature-optimization/`
**Prerequisites**: plan.md, spec.md
**Workflow**: Ephemeral Testing (Sandbox -> Verify -> Integrate) per User Story

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Parallelizable task
- **[Story]**: User Story ID (US1, US2, US3)

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Initialize the ephemeral development environment.

- [x] T001 Create `testing.py` with SparkSession setup and `load_data` from `data_transformation.py`

## Phase 2: User Story 1 - Robust Outlier Management (Priority: P1)

**Goal**: Implement dynamic IQR-based price outlier filtering.
**Independent Test**: Verify listings > Q3 + 1.5*IQR are removed.

- [ ] T002 [US1] Implement `calculate_outlier_bounds` using `approxQuantile` in `testing.py`
- [ ] T003 [US1] Implement filtering logic using calculated bounds in `testing.py`
- [ ] T004 [US1] Verify outlier removal by checking max price and row counts in `testing.py`
- [ ] T005 [US1] Migrate outlier logic to **Definitions** section in `data_transformation.py`
- [ ] T006 [US1] Integrate outlier filter step into `apply_stateless_transformations` in `data_transformation.py`

## Phase 3: User Story 2 - Logical Imputation (Priority: P2)

**Goal**: Impute structural missing values with 0.
**Independent Test**: Verify nulls in beds/bedrooms/baths are 0.

- [ ] T007 [US2] Implement structural imputation (`fillna(0)`) for beds, bedrooms, baths in `testing.py`
- [ ] T008 [US2] Verify zero nulls in structural columns in `testing.py`
- [ ] T009 [US2] Migrate structural imputation logic to **Definitions** section in `data_transformation.py`
- [ ] T010 [US2] Integrate structural imputation step into pipeline in `data_transformation.py`

## Phase 4: User Story 3 - Richer Feature Engineering (Priority: P2)

**Goal**: Extract amenity groups and host longevity metrics.
**Independent Test**: Verify new score columns and longevity calculations.

- [ ] T011 [P] [US3] Implement `host_longevity` calculation (`datediff`) and response rate imputation (`median`) in `testing.py`
- [ ] T012 [P] [US3] Implement amenity parsing and scoring for fixed groups (Essentials, Luxury, Security, Family, Pet) in `testing.py`
- [ ] T013 [US3] Verify new columns (`amenities_luxury_score`, `host_longevity_days`, etc.) in `testing.py`
- [ ] T014 [US3] Migrate feature engineering functions to **Definitions** section in `data_transformation.py`
- [ ] T015 [US3] Integrate feature engineering steps into pipeline in `data_transformation.py`
- [ ] T016 [US3] Update `fit_transform_features` in `data_transformation.py` to include new features in VectorAssembler

## Phase 5: Polish & Cleanup

**Purpose**: Final system validation and constitution compliance.

- [ ] T017 Run full `data_transformation.py` execution
- [ ] T018 Validate output schema matches `contracts/schema.md`
- [ ] T019 **DELETE** `testing.py` (Constitution Requirement)

## Dependencies & Execution Order

1.  **Setup**: T001 blocks everything.
2.  **US1 (Outliers)**: T002-T006 must complete to ensure clean data for subsequent features.
3.  **US2 & US3**: Can theoretically run in parallel, but sequential integration into `data_transformation.py` is safer to avoid merge conflicts in the single file.
4.  **Cleanup**: Must run last.

## Implementation Strategy

1.  **MVP**: Complete Phase 1 & 2 (Outlier cleaning).
2.  **Increment 1**: Add Phase 3 (Structural Imputation).
3.  **Increment 2**: Add Phase 4 (Feature Engineering).
4.  **Finalize**: Run Phase 5.
