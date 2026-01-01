# Future Data Augmentation & Features

The following methodologies were identified in the source projects but cannot be currently implemented due to missing data columns in the primary dataset (`airbnb.csv`). These would require additional scraping or data enrichment.

## 1. Property Age & History
- **Construction Year:** (`sources/project_3`)
  - **Concept:** Used as a proxy for property modernity and age.
  - **Method:** MinMax scaling of the construction year.
  - **Status:** The current dataset does not contain a `construction_year` or `built_date` column.
