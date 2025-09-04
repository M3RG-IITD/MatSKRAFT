# Linking the Extracted Information and Final Evaluation

Merges predicted compositions and properties into a unified, machine-readable database and computes **official evaluation metrics** at the integrated level.  
Final performance is benchmarked against a gold standard constructed via expert annotation.


## Overview

The workflow proceeds in three phases:

1. Evaluate property and composition tuples individually.
2. Integrate them using intra and inter-table alignment via **table orienation and material IDs respectively**.
3. Score predictions against gold database to obtain **final metrics**.


## Follow the Notebooks in Order

```
1_property_extraction_results.ipynb
```

Evaluates property prediction tuples.

**Scores (Before Linking):**

| Metric    | Precision  | Recall    | F1-score  |
|-----------|------------|-----------|-----------|
| Score     | **90.35%** | **87.07%**| **88.68%** |

```
2_composition_extraction_results_and_materials.ipynb
```
**Scores (Before Linking):**

| Metric    | Precision  | Recall    | F1-score   |
|-----------|------------|-----------|------------|
| Score     | **82.31%** | **62.97%**| **71.35%** |

```
3_detecting_orientation.ipynb
```

Infers the row-/column-major orientation for each table using prediction maps.  
This is critical for deciding how composition and properties should be aligned within the table.

```
4_preparing_res_prop.ipynb
```
Cleans and harmonizes predicted property results:  
standardizes value formats, filters noise, and attaches metadata needed for linking.

```
5_constructing_the_predicted_database.ipynb
```
Constructs the **predicted database** by linking compositions and properties using orientation and proxy IDs.

**Key steps:**
- **Intra-table linking:** Matches compositions and properties on shared row/column axes based on orientation.
- **Proxy ID generation:** Forms unique IDs, ensuring traceability back to the source and act as the universal linking key.
- **Inter-table linking:** Connects related tables using proxy IDs on shared material identifiers.

**Output:**  
A unified prediction database of composition–property tuples with structural coherence and traceability.

```
6_constructing_the_gold_database.ipynb
```
Creates the **gold standard database** from expert-annotated data, mirroring the prediction schema.  
Used as the reference for final evaluation.

```
7_final_score_final.ipynb
```

Computes official end-to-end metrics after database integration.

**Final Scores:**

| Task                      | Precision | Recall | F1     |
|---------------------------|----------:|-------:|-------:|
| **Property Extraction**   | 91.22%    | 87.37% | 89.26% |
| **Composition Extraction**| 81.47%    | 66.48% | 73.22% |
| **Integrated Database**   | **78.08%**| **61.26%** | **68.66%** |

> **Note:** Post-linking scores include additional integration checks during linking that enhance consistency and slightly improve performance. For transparency, the article reports **pre-linking scores** obtained from the property and composition modules, while this repository presents both the pre-linked and the **post-linked results** for transparency.



## Output Schema

Each record in the integrated database (`database_pred.pkl`) follows:

```python
[
  paper_id,                 # unique article identifier
  [table_index],            # table index
  material_id,              # material identifier if available
  proxy_id,                 # universal linking key 
  composition_tuples_sorted,# list of constituents [(element1, quantity, unit), (element2, quantity, unit), ...]  tuples
  property_tuple,           # (property_name, value, unit)
  journal_name,             # journal to which the article belongs
]
```


