# Training Data Generation

This module provides the data preparation pipelines for training the two core components of **MatSKRaFT**:  
- **Property_extraction** – 3-stage hierarchical data preparation strategy for extracting properties
- **Composition_extraction** – 2-stage data preparation strategy for extracting compositions

The pipelines extend beyond distant supervision by integrating annotation algorithms, chemical reasoning, and targeted augmentation strategies, thereby producing datasets that are chemically consistent, structurally diverse, and reflective of real-world reporting practices. Models trained on these datasets are rigorously validated against expertly annotated development and test sets, where each component of the pipeline proves essential - discussed in details in the Ablation Stuides.

- The **[Property_extraction](./Property_extraction)** workflow builds labeled property datasets using distant supervision, annotation algorithms, and strategic augmentation routines. These steps normalize units, resolve ambiguities, expand coverage of underrepresented properties, and output clean property–value–unit tuples suitable for training.  

- The **[Composition_extraction](./Composition_extraction)** workflow generates labeled composition datasets. It combines distant supervision with multi-layered annotation algorithms, which enhances the extraction accuracy of compositions from material science tables.

Together, these pipelines yield robust training corpora that enable MatSKRAFT to achieve broad coverage and high extraction accuracy across diverse notations and reporting conventions, while eliminating the reliance on costly manual annotation typically required in specialized scientific domains.
