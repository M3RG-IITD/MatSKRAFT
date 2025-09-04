# MatSKRAFT - Property Extraction Module

This folder contains the code and data for the **property extraction component** of our paper [MatSKRAFT — A framework for large-scale materials knowledge extraction from scientific tables](https://arxiv.org/abs/XXXX.XXXXX).

---
## Installing Requirements
```
bash install_requirements.sh
```

## Overview

The property extraction module of **MatSKRAFT** is designed to automatically identify, normalize, and structure **materials property values for the 18 targeted properties** reported in material science tables.  
It leverages a **Graph Neural Network (GNN)** model combined with **constrained-learning**, domain-specific **unit handling**, and **post-processing rules** to ensure high precision in real-world materials science datasets.

### Key features
- **Graph-based modeling**: Each table is represented as a graph, enabling the model to capture structural and contextual dependencies.  
- **Unit normalization**: Dedicated modules standardize heterogeneous unit expressions into canonical forms.  
- **Post-processing pipelines**: Rule-based refinement safeguards scientific consistency, correcting mispredictions and filtering out invalid entries.  
- **Wide property coverage**: Extracts a **broad spectrum of 18 material properties** spanning physical, mechanical, optical, and electrical domains.  
- **End-to-end integration**: From raw table parsing to machine-readable structured output, the module is fully automated.

---

## Properties Covered

MatSKRAFT is designed to handle both **well-studied** and **rare, long-tail properties**.  
Our system successfully extracts the following 18 key properties:

- **Physical**: Activation energy, Annealing point, Crystallization temperature, Glass transition temperature, Liquidus temperature, Melting temperature, Softening point, Thermal expansion coefficient.  
- **Mechanical**: Bulk modulus, Density, Fracture toughness, Hardness, Poisson ratio, Shear modulus, Young’s modulus.  
- **Optical**: Abbe number, Refractive index.  
- **Electrical**: Electrical conductivity.  


This wide coverage ensures that the resulting database captures **both critical design properties** (e.g., modulus, toughness) and **functional properties** (e.g., conductivity, refractive index), enabling a diverse set of materials informatics applications.

---

## Workflow

The property extraction pipeline involves the following stages:

1. **Table Parsing** — convert tables into graph-structured representations.  
2. **Model Training** — run the GNN model (`gnn_model.py`) to learn property–value–unit associations.  
3. **Unit Handling** — extract units using `units.py` and `units_2.py`.  
4. **Post-Processing** — refine predictions with `post_processing.py` and `post_processing_2.py`.  
5. **Structured Output** — produce machine-readable property datasets for integration into the knowledge base.

---

## Training

To train the property extraction model:

```bash
bash run.sh 2 #2 - seed



## Modules executed by run.sh

## train_gnn.py

Inside train_gnn.py:
from gnn_model import GNN_Model as Model
from utils import *
from units import *
from post_processing import *
from post_processing_2 import *

```

## Inferening

Inferencing is performed with:

```bash
bash run_test_2_layer_sep_splits_inference.sh
```

<!-- ## Results

MatSKRAFT’s property extraction achieves **state-of-the-art performance**:

- **Precision**: 90.35, **Recall**: 87.07, **F1 score**: 88.68   -->
