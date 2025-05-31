# Data Cleaning and Bias Calculation 

## `prompt_bias_clean_cal_update.ipynb`

This notebook performs parallel processing of text inputs (like IAT stimuli) to compute bias scores based on predefined word pairings and valence associations.
dask.config.set(scheduler='distributed')
cluster = SLURMCluster(...)
client = Client(cluster)
Using dask ensures the workload is distributed across compute nodes effectively. Dask is especially useful when working with pandas data leaning and algebra between columns.

### Key Functions
#### `extract_word_pairs(text, group_word)`
- Parses a string to extract word pairs.
- Handles different formats (e.g., `1. happy - gay`, `happy gay`).
- **Returns:** list of `(valence, group)` tuples.
#### `d_score(a,b,c,d)`
- Computes a composite bias score:
  \[
  D = \frac{a}{a+b} + \frac{d}{c+d} - 1
  \]
  where \(a\), \(b\), \(c\), \(d\) are counts of combinations of valence (`positive`/`negative`) and group (`stigma`/`default`).
- **Returns:** score in [-1, 1].
- Using pycc to compile this function can slightly increase the conputing speed.
#### `process_iat_data_parallel(input_file, label_file, output_file)`
- Main routine that:
  - Reads raw text and label files.
  - Applies `process_entry` in parallel using Dask.
  - Writes final results (bias scores and text content) to CSV.
#### `process_entry(entry)`
- Helper function for parallel execution.
- Extracts word pairs from a single text, labels them, and calculates the bias score using `d_score`.
#### `preprocess_with_slurm`
- Helper function for parallel execution.
- clean out iat word pairs.

- **Returns:** `(iat)`.
- This function is designed to operate independently on a single row of input. Each call to process_entry() is wrapped using dask.delayed() to build a computation graph instead of executing immediately.
- The entire list of delayed tasks is executed in parallel using dask.compute(), which leverages SLURM-managed resources on HPC:

---

# Correlation Analysis

## `power_correlation_visual_pipeline.ipynb`
This repository contains a pipeline (power_correlation_visual_pipeline.ipynb) for analyzing implicit bias across transformer model layers. The pipeline compares model-generated bias scores with human-annotated outcomes such as IAT bias and report scores, calculates correlations, builds statistical models, and generates comprehensive visualizations.
### Large-Scale Computing Strategy
There are several statistic variables to calculate but sharing similar computing strategy

#### `Parallel Computation`
This part used SLURM-based high-performance clusters using dask_jobqueue.SLURMCluster. All layer-specific computations (e.g., correlations, regressions) are delayed with dask.delayed. Results are computed concurrently using dask.compute().Similarly dask.bag is used to process repeated operations over sequences.
