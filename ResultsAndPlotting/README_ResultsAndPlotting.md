# Results and Plotting Module

This directory contains all the analysis, results processing, and visualization components for the DL4NLP project.

## Directory Structure

```
ResultsAndPlotting/
├── analysis/                          # Analysis notebooks and helper functions
│   ├── analysis.ipynb                 # Main analysis notebook
│   ├── flores_101_metadata.csv       # FLORES-101 dataset metadata
│   └── helpers.py                     # Analysis helper functions
├── outputs/                           # All experimental results and processed data
│   ├── collected_results/             # Aggregated results
│   ├── flores_eval/                   # FLORES evaluation results
│   ├── multilang_eval/                # Multi-language evaluation results
│   ├── pruning_WMT/                   # Pruning experiment results
│   ├── WMT_multiling_eval/           # WMT multi-language evaluation
│   ├── wmt24pp_eval/                  # WMT24++ evaluation results
│   └── results_analysis.ipynb        # Results analysis notebook
├── tables_plotting/                   # Core plotting and visualization module
│   ├── __init__.py
│   ├── helpers.py                     # Plotting helper functions
│   ├── plotting.py                    # Main plotting functions
│   ├── result_processing.py           # Data processing utilities
│   └── plots/                         # Generated plots and visualizations
├── flores_plus/                       # FLORES+ related files
├── direction_analysis_demo.py         # Direction analysis demonstration script
├── test_direction_analysis.py         # Tests for direction analysis
├── DIRECTION_ANALYSIS_README.md       # Direction analysis documentation
└── README.md                         # Original project README

```

## Main Components

### 1. Analysis Module (`analysis/`)
- Interactive Jupyter notebooks for data exploration
- Helper functions for statistical analysis
- Metadata files for datasets

### 2. Results Storage (`outputs/`)
- Organized by evaluation type and experiment
- CSV files with evaluation metrics
- Raw output files from model evaluations
- Analysis notebooks with processed results

### 3. Visualization Module (`tables_plotting/`)
- Core plotting functionality
- Data processing utilities
- Generated visualization outputs
- Support for multiple plot types (heatmaps, line charts, bar charts)

### 4. Direction Analysis
- Specialized analysis for translation direction effects
- Demonstration scripts and tests
- Documentation for direction-specific metrics

## Usage

This module is designed to be integrated into the main DL4NLP project while maintaining clear separation of concerns for results processing and visualization.

## Integration Notes

When merging with the main branch:
- This entire `ResultsAndPlotting/` directory can be added to the main project
- Dependencies and imports may need to be updated to reflect the new structure
- Consider adding this module to the main project's setup.py or requirements.txt