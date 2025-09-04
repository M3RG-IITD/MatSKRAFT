# Baseline Comparison with Large Language Models

To benchmark MatSKRaFT against state-of-the-art LLMs, we designed prompt-based extraction pipelines for both **property** and **composition** tasks.  
These notebooks replicate the experiments reported in **Section 2.3.3 (Benchmarking Against Large Language Models)**, **Appendix C (Performance Benchmarking and Computational Efficiency)**, and follow the optimization principles detailed in **Appendix D (Comprehensive LLM Optimization and Fair Evaluation Framework)** of the paper.

## Overview

- **Property extraction baseline**  
  Implemented in `property_baseline.ipynb` and evaluated via `check_scores_property_baselines.ipynb`.  

- **Composition extraction baseline**  
  Implemented in `composition_baseline.ipynb` and evaluated by `check_scores_composition_baselines.ipynb`.

  ![MatSKRAFT](Figure_2.pdf)
  
  Both baseline comparisons use optimized few-shot prompts enriched with textual context, carefully tuned on a small validation set, and include parser-aware safeguards, auto-fallbacks for malformed generations, and multiple-retry mechanisms for rate-limit or generation failures. These baselines evaluate LLM performance on the same held-out test splits used by MatSKRaFT, ensuring direct comparability.  
---
