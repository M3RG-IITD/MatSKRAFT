# Baseline Comparison with Large Language Models

To benchmark MatSKRaFT against state-of-the-art LLMs, we designed prompt-based extraction pipelines for both **property** and **composition** tasks.  
These notebooks replicate the experiments reported in **Section 2.3.3 (Benchmarking Against Large Language Models)**, **Appendix C (Performance Benchmarking and Computational Efficiency)**, and follow the optimization principles detailed in **Appendix D (Comprehensive LLM Optimization and Fair Evaluation Framework)** of the paper.

## Overview

- **Property extraction baseline**  
  Implemented in `property_baseline.ipynb` and evaluated via `check_scores_property_baselines.ipynb`.  

- **Composition extraction baseline**  
  Implemented in `composition_baseline.ipynb` and evaluated by `check_scores_composition_baselines.ipynb`.  
  
  Both baseline comparisons use optimized few-shot prompts enriched with textual context, carefully tuned on a small validation set, and include parser-aware safeguards, auto-fallbacks for malformed generations, and multiple-retry mechanisms for rate-limit or generation failures. These baselines evaluate LLM performance on the same held-out test splits used by MatSKRaFT, ensuring direct comparability.  
---

## Performance Comparison with Large Language Models
<table>
  <thead>
    <tr>
      <th rowspan="2">Models</th>
      <th colspan="4">Extracting Properties</th>
      <th colspan="4">Extracting Composition</th>
    </tr>
    <tr>
      <th>Precision</th>
      <th>Recall</th>
      <th>F1 Score</th>
      <th>Time (s/table)</th>
      <th>Precision</th>
      <th>Recall</th>
      <th>F1 Score</th>
      <th>Time (s/table)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Claude-3.5-Sonnet</td>
      <td>78.80</td><td>67.64</td><td>72.79</td><td>4.90</td>
      <td>47.12</td><td>55.38</td><td>50.92</td><td>6.56</td>
    </tr>
    <tr>
      <td>DeepSeek-R1</td>
      <td>70.43</td><td>68.07</td><td>69.23</td><td>114.94</td>
      <td>52.85</td><td>55.79</td><td>54.28</td><td>187.59</td>
    </tr>
    <tr>
      <td>DeepSeek-V3</td>
      <td>75.17</td><td>71.48</td><td>73.28</td><td>14.95</td>
      <td>49.71</td><td>52.26</td><td>50.95</td><td>18.07</td>
    </tr>
    <tr>
      <td>Gemini-1.5-Pro</td>
      <td>66.09</td><td>58.13</td><td>61.85</td><td>8.25</td>
      <td>39.60</td><td>44.02</td><td>41.69</td><td>11.89</td>
    </tr>
    <tr>
      <td>GPT-4o</td>
      <td>61.61</td><td>59.03</td><td>60.29</td><td>5.84</td>
      <td>47.46</td><td>53.42</td><td>50.27</td><td>9.30</td>
    </tr>
    <tr style="background-color:#ffe6f0; font-weight:bold;">
      <td>MatSKRaFT</td>
      <td>90.35</td><td>87.07</td><td>88.68</td><td>0.22</td>
      <td>82.31</td><td>62.97</td><td>71.35</td><td>0.39</td>
    </tr>
  </tbody>
</table>

<p><b>Table:</b> Performance comparison of <b>MatSKRaFT</b> against large language models for materials science information extraction from tables.  
Time measurements represent average processing duration per table.</p>
