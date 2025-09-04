# Ablation Studies

We quantify the contribution of each component using **single-factor** and **combined** ablations on the test dataset (see Appendix B).  
Each variant is trained and evaluated on the same splits; metrics are **F1 / Precision / Recall**.

## Property Extraction (MatSKRaFT_P)

| Model Configuration                              | F1    | Precision | Recall |
|--------------------------------------------------|------:|----------:|-------:|
| **MatSKRaFT_P (full)**                           | **88.68** | 90.35     | 87.07  |
| w/o Constrained learning*                        | 88.38 | 88.04     | 88.72  |
| w/o Caption information                          | 86.94 | 85.94     | 87.95  |
| w/o Post-processing                              | 79.30 | 76.63     | 82.16  |
| w/o Data augmentation                            | 87.50 | 88.37     | 86.65  |
| w/o Annotation codes                             | 79.66 | 72.34     | 88.64  |
| w/o Distant supervision                          | 86.62 | **93.04** | 81.02  |
| w/o (Distant supervision + Data augmentation)    | 86.20 | 91.17     | 81.11  |
| w/o (Annotation codes + Data augmentation)       | 84.31 | 80.09     | **89.01** |

\* ID constraint improved the **ID classification F1 in property tables** from **81.4% → 82.9%**.

**Observation :** The largest drops come from removing **Post-processing** (−9.38 F1) and **Annotation codes** (−9.02 F1). Caption context and data augmentation also provide consistent gains.


**Using the logs** : Each ablation run writes a plain-text summary (e.g., `ablation_*.txt`) with the metrics obtained while training the ablation models.  

---

## Composition Extraction (MatSKRaFT_C)

| Model Configuration              | F1    | Precision | Recall |
|----------------------------------|------:|----------:|-------:|
| **MatSKRaFT_C (full)**           | **71.35** | **82.31** | **62.97** |
| w/o Thresholding                 | 68.73 | 78.19     | 61.32  |
| w/o Constrained learning         | 64.83 | 76.86     | 56.05  |
| w/o Annotation codes             | 62.42 | 73.57     | 54.21  |
| w/o Caption information          | 61.64 | 73.12     | 53.27  |

**Observation :** The largest drop is from removing **Caption information** (−9.71 F1), followed by **Annotation codes** and **Constrained learning**. Thresholding contributes measurable stability.

```
bash compute_combined_results.sh
```
On executing the above files, one can verify all the ablation studies performed on the composition extraction component.

## 

---




