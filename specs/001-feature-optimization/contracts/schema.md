# Data Contract: Transformed Output

**Type**: DataFrame Schema
**Source**: `data_transformation.py` output

## Output Columns

The final DataFrame used for model training MUST contain the following columns (in addition to existing ones):

```yaml
columns:
  - name: price_cleaned
    type: DoubleType
    nullable: false
    description: Log1p transformed price, filtered using dynamic IQR (Q3 + 1.5*IQR)

  - name: amenities_security_score
    type: IntegerType
    nullable: false
    description: Count of security items (Lockbox, Smart Lock, etc.)

  - name: amenities_luxury_score
    type: IntegerType
    nullable: false
    description: Count of luxury items (Pool, Hot tub, etc.)

  - name: amenities_essentials_score
    type: IntegerType
    nullable: false
    description: Count of essential items

  - name: amenities_family_score
    type: IntegerType
    nullable: false
    description: Count of family-friendly items

  - name: amenities_pet_score
    type: IntegerType
    nullable: false
    description: Count of pet-friendly items

  - name: host_longevity_days
    type: IntegerType
    nullable: true
    description: Days between host_since and current_date

  - name: num_bedrooms
    type: DoubleType
    nullable: false
    description: Structural count (imputed with 0.0)

  - name: num_beds
    type: DoubleType
    nullable: false
    description: Structural count (imputed with 0.0)

  - name: num_baths
    type: DoubleType
    nullable: false
    description: Structural count (imputed with 0.0)
```

## Validation Rules

1.  **No Nulls**: `num_bedrooms`, `num_beds`, `num_baths` must have 0 nulls.
2.  **Price Range**: `exp(price_cleaned) - 1` must be <= Q3 + 1.5 * IQR (calculated from training distribution).
3.  **Data Types**: All feature columns must be numeric (Double/Integer) for the VectorAssembler.