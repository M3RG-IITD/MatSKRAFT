# Training Data Generation

## For property extraction

To train **MatSKRaFT_P** (property extractor), we construct labeled datasets through a **multi-stage annotation pipeline**.  

### Steps to Annotate the Training Dataset

#### *Follow the notebooks **in order** (pre-augmentation):*

```bash
1_create_combined_database.ipynb
```

This step filters the database to retain only the targeted set of material properties by merging raw tabular data with INTERGLAD reference entries, based on property identifiers.  


```
2_distant_supervision_kausik_max.ipynb
```

Aligns table values with INTERGLAD entries to generate the initial distant-supervised label set.

**Key steps:**
- **Numeric parsing:** Extracts all numbers from table cells for each target property.
- **Property-wise matching:** Performs exact and tolerance-based alignment with database values using property-specific thresholds.
- **Unit normalization:** Applies property-specific rules — e.g., infers `g·cm⁻³` vs `kg·m⁻³` for Density based on magnitude and enforces one consistent transform (×1000/÷1000) per table; reconciles Celsius/Kelvin for Tg by testing (x, x−273, x+273); and matches dimensionless properties (e.g., refractive index, Abbe) directly.
- **Match capping:** Limits per-value matches to avoid false positives.
- **Orientation inference:** Declares a table as property-bearing if ≥30% of values align on any axis, and infers whether property varies across rows or columns.
- **Tuple generation:** Emits `(db_index, value, unit)` tuples from identified property-bearing rows/columns.

**Output:**  
A wide-coverage set of labeled property values matched to INTERGLAD, filtered by orientation and unit consistency — serving as the initial training signal.




```
3_check_after_ds_after_choosing_property.ipynb
```


Applies property-specific annotation algorithms to **expand and disambiguate** labels beyond distant supervision using scientific reasoning and multi-criteria matching on the tables which weren't annotated using distant supervision.



**Key steps:**
- **Ambiguity resolution:** Resolves overloaded symbols (e.g., `$n$` as refractive index vs. Poisson’s ratio) via multi-factor verification (units, values, caption cues).
- **Header detection:** Identifies property rows/columns using expanded aliases (e.g., `Tg`, `T_glass`, `glass point`) and robust unit-aware regexes.
- **Structure scoring:** Quantifies alignment strength across rows/columns and assigns property orientation accordingly.
- **Fallback recovery:** Captures low-signal cases (e.g., conductivity, activation energy) using weak headers or unit-only matches.

**Output:**  
An expanded and disambiguated set of property annotations with significantly improved coverage — resulting high precision-labelled dataset.

```
4_prop_before_final.ipynb
```

Applies annotation alogithms on the columns/rows of the tables which are classified as non-property, but atleast one property column/row has been annotated by distant supervision. Also performs post-annotation cleanup on the distantly-supervised labeled property tables to ensure consistency by deduplication, orientation normalization, and cleanup for consistency enforcement.

```
5_add_keys_to_string_matched_list_&_create_combined_labelled_train.ipynb
```
Merges distantly-supervised and annotation-code-labelled tables into a unified training set with key-based indexing.

```
6_check_for_units.ipynb
```
This module implements unit detection for material properties, handling diverse notation systems and ambiguous representations commonly found in scientific literature.

```
7_check_normalization.ipynb
```

Automated unit normalization system that standardizes diverse unit representations across the desired 18 material properties, mapping the vast notation variants to canonical forms with property-specific validation rules. The system processes property-value-unit tuples from extracted tables and updates the unit field instead of different semantic phrases representing the same unit.

```
8_add_non_prop.ipynb
```
Training data preparation module that adds non-property tables from articles containing at least one property table to provide negative examples. This ensures the model learns to distinguish between property-bearing and non-property tables by exposing it to both positive and negative cases during training.

```
9_add_ids_from_manual_annotation.ipynb
```
Manual annotation module for labelling material identifiers in tables, enabling cross-table linking of compositions and properties that refer to the same materials. This annotation supports the dual-pathway integration approach by providing material IDs necessary for connecting related information across separate tables within publications.

```
10_tuple_generation.ipynb
```

Tuple generation module that converts extracted property data into standardized (ID, property_name, value, unit) tuples for downstreaming prediction task and knowledge-base construction. The ID, unique to each extraction, is constructed by combining publication info, table position, positional indices, and material IDs to enable traceability and cross-table linking of related information.

```
11_modify_data.ipynb
```
Minor data modification module that consolidates duplicate property labels by merging "Softening Point (Viscosity)" with "Softening Point (Temperature)" and adjusts all subsequent label indices. This ensures consistent property representation by combining related softening point measurements under a single standardized label.

#### *Follow the notebooks **in order** (for augmentation):*

```
1_plan_for_augmentation.ipynb
```
Analyzes existing label frequency across the 18 properties to design an augmentation strategy that addresses long-tail underrepresentation.


```
2_rough_changes.ipynb
```

Initializes augmentation-tracking keys in every table to record which specific rows and columns are augmented — enabling clean separation of real and synthetic data in downstream processing.


```
3_do_aumentation.ipynb
```
Implements statistically guided augmentation by injecting synthetic tuples into real tables — strategically amplifying underrepresented properties based on co-occurrence patterns with frequently reported "neighbor" properties to mimic real material tables for future use.

**Key steps:**
- **Neighborhood-driven injection:** Identifies co-studied property pairs (e.g., Abbe value with refractive index) and augments rare properties into real tables containing their neighbors.
- **Power-law scaling:** Computes property-specific augmentation factors $n_{\text{new}} = \lceil a \cdot n_{\text{original}}^{\alpha} \rceil$ ($a=10$, $\alpha=0.65$) to balance coverage while preserving natural frequency hierarchies.
- **Synthetic value generation:** Samples from Gaussian $\mathcal{N}(\mu, \sigma)$ distributions fit to source property values, clipped to $\pm 3\sigma$ to ensure physical realism.
- **Table reconciliation:** Aligns source and target column lengths by sampling or clipping values for structural consistency in augmented tables.
- **Augmentation tracking & safeguards:** Marks modified rows/columns for traceability; skips tables with no valid numeric values; injects controlled noise when $\sigma = 0$; discards out-of-range samples.

**Output:**  
A balanced, physically plausible training dataset enriched with statistically consistent synthetic tuples — significantly expanding coverage of rare materials properties while preserving the natural frequency distribution of the original data (see Figure A.1 in article).

![training_data_stats](../train_data_generation/ds_anno_aug_stats.png)

#### Post augmentation

Follows the notebook sequence to finalize clean, unit-consistent tuples for downstream training:

```
add_keys_to_whole_data.ipynb → check_for_units.ipynb → check_normalization.ipynb → tuple_generation.ipynb
```

These steps:
- Load augmented datasets  
- Validate and normalize units  
- Ensure property-specific consistency  
- Emit final training tuples in the required schema  

---





