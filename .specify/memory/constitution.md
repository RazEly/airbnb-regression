<!--
Sync Impact Report:
- Version change: 0.0.0 -> 1.0.0
- List of modified principles:
  - Single File Architecture (New)
  - Notebook-Style Structure (New)
  - Logical Flow (New)
  - Ephemeral Testing (New)
- Added sections: None
- Removed sections: None
- Templates requiring updates:
  - .specify/templates/tasks-template.md (✅ updated: reflects testing.py workflow)
  - .specify/templates/plan-template.md (✅ compatible: generic Constitution Check gate)
- Follow-up TODOs: None
-->

# Airbnb Data Transformation Constitution

## Core Principles

### I. Single File Architecture
The project must contain a single production file named `data_transformation.py`. No other code files are permitted for production use. This file serves as the sole artifact for the transformation pipeline.

### II. Notebook-Style Structure
`data_transformation.py` must be structured like a Jupyter notebook using the `py:percent` convention (e.g., `# %%`). This ensures compatibility with interactive execution environments while maintaining a script format.

### III. Logical Flow
The code must follow a strict logical order:
1. **Definitions**: Functions for transformations and feature engineering.
2. **Pipeline**: Machine Learning pipeline execution.
3. **Training**: Model training and evaluation.

### IV. Ephemeral Testing Workflow
All transformation features must be developed and tested in a separate `testing.py` file. This file is temporary and exists only during the development of a specific feature. Once the feature is verified, it is integrated into the main `data_transformation.py` file, and `testing.py` MUST be deleted.

### V. No Production Clutter
Files other than `data_transformation.py` (and temporary `testing.py`) are strictly prohibited in the production path. The repository must remain clean of auxiliary scripts or unused modules.

## Project Constraints

### Technology Stack
- **Language**: Python
- **Format**: `py:percent` (Jupyter-compatible script)
- **Primary File**: `data_transformation.py`

## Development Workflow

### Feature Implementation Cycle
1. Create `testing.py`.
2. Implement and test the new feature or transformation in `testing.py`.
3. Verify correctness.
4. Integrate the code into the appropriate section (Definitions, Pipeline, or Training) of `data_transformation.py`.
5. Run `data_transformation.py` to ensure full system integrity.
6. Delete `testing.py`.

## Governance

### Amendment Procedure
Amendments to this constitution require a version bump. Changes to the "Single File Architecture" or "Ephemeral Testing Workflow" are considered Major changes (X.0.0).

### Compliance
All code contributions must be reviewed against these principles. Any PR containing persistent files other than `data_transformation.py` (e.g., leftover `testing.py` or utils) must be rejected.

**Version**: 1.0.0 | **Ratified**: 2026-01-01 | **Last Amended**: 2026-01-01