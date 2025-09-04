## For composition extraction

To train **MatSKRaFT_C** (composition extractor), we construct labeled datasets through a **multi-stage annotation pipeline**.  
This extends the DiSCoMaT distant supervision workflow with an additional **Annotation Codes** step that injects chemical reasoning and validation.

---

### Steps to Annotate the Training Dataset

```bash
python distant_supervision_scc.py
```
Reads train_data_new.pkl and SciGlass database files to annotate SCC tables (single-composition cells).
Output: train_data_scc.pkl

```
python distant_supervision_mcc_ci.py
```
Reads train_data_scc.pkl and SciGlass files to annotate MCC tables (multi-cell composition tables).
Output: train_data_mcc_ci.pkl

```
python distant_supervision_mcc_pi.py
```
Reads train_data_mcc_ci.pkl, SciGlass files, and train_val_test_paper_data.pkl to annotate MCC–PI tables (multi-composition with property-integrated structure).
Output: train_data_mcc_pi.pkl

```
python modify_train_data.py
```

Applies Annotation Codes to enrich the distantly-supervised datasets. This implements a multi-layered chemical reasoning system.

- Formula recognition across diverse notation systems.  
- Stoichiometric validation ensuring charge balance and chemical plausibility, with percentage-sum checks (valid totals in the 95–105% range) and handling of residuals as dopants/error margins.  
- Unit detection and normalization of composition conventions (mol%, wt%, at%).  
- False positive filtering to eliminate non-compositional values.  
- Coverage expansion beyond distant supervision, systematically labeling compositional rows and columns absent in database matches, increasing both the quantity and quality of training examples.  

Final Output: a chemically validated and structurally diverse dataset, essential for training MatSKRaFT_C with robust coverage of real-world reporting formats.
