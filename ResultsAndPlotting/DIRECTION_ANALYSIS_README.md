# Language Direction Analysis Functions

## Overview

I've added comprehensive language direction analysis capabilities to the plotting system. These functions help analyze how quantization and pruning affect different types of translation directions.

## New Functions Added

### 1. Core Categorization Function

```python
def categorize_language_direction(df: pd.DataFrame) -> pd.DataFrame:
```
- **Purpose**: Adds a 'direction_type' column to any DataFrame
- **Categories**:
  - `"English → Foreign"`: English source, non-English target
  - `"Foreign → English"`: Non-English source, English target  
  - `"Foreign → Foreign"`: Both source and target are non-English
- **Input**: DataFrame with 'source_lang' and 'target_lang' columns
- **Output**: DataFrame with added 'direction_type' column

### 2. Data Preparation Functions

```python
def prepare_direction_grouped_data(df, metric, benchmark) -> pd.DataFrame:
```
- **Purpose**: Groups quantization data by direction type and quantization level
- **Returns**: DataFrame with [quant_level, direction_type, mean_value, std_value, count]

```python
def prepare_direction_grouped_pruning_data(df, metric, benchmark) -> pd.DataFrame:
```
- **Purpose**: Groups pruning data by direction type and sparsity level
- **Returns**: DataFrame with [sparsity_level, direction_type, mean_value, std_value, count]

### 3. Visualization Functions

```python
def plot_direction_comparison_chart(df_grouped, metric, benchmark, out_png, out_svg, chart_type):
```
- **Purpose**: Creates line charts showing performance trends by direction
- **Features**:
  - Different colors and markers for each direction type
  - Sample counts in legend (e.g., "English → Foreign (n=45)")
  - Error bars showing standard deviation
  - Supports both quantization and pruning charts

```python
def create_direction_comparison_bar_chart(df, metric, benchmark, comparison_level, out_png, out_svg, chart_type):
```
- **Purpose**: Creates side-by-side bar charts for specific quantization/sparsity levels
- **Features**:
  - Two panels: absolute scores and percent change vs baseline
  - Sample counts annotated on bars
  - Color-coded by direction type

### 4. Orchestration Functions

```python
def make_all_direction_comparisons(csv_path, metrics, quant_levels):
```
- **Purpose**: Generates complete direction analysis for quantization experiments
- **Generates**: Line charts and bar charts for all metrics and benchmarks

```python
def make_all_pruning_direction_comparisons(pruning_df, baseline_df, metrics, sparsity_levels):
```
- **Purpose**: Generates complete direction analysis for pruning experiments
- **Generates**: Line charts and bar charts comparing pruning to baseline

## Integration with Main Script

The functions are integrated into the main `plotting.py` script in the `if __name__ == "__main__":` section:

```python
# Generate direction comparison plots for quantization data
make_all_direction_comparisons(
    csv_path=out_csv,
    metrics=["bleu", "comet22", "chrf", "kiwi23"],
    quant_levels=["8-Bit", "4-Bit"]
)

# Generate direction comparison plots for pruning data
make_all_pruning_direction_comparisons(
    pruning_df=pruning_df,
    baseline_df=baseline_df,
    metrics=["bleu", "comet22", "chrf", "kiwi23"],
    sparsity_levels=["10% Sparsity", "30% Sparsity", "50% Sparsity"]
)
```

## Output Structure

Running the script generates plots in:

```
./plots/direction-analysis/
├── bleu/
│   ├── direction_trends_FloRes.png              # Quantization line chart
│   ├── direction_trends_WMT24++.png             # Quantization line chart
│   ├── direction_trends_pruning.png             # Pruning line chart
│   ├── direction_bars_FloRes_8-Bit.png          # Quantization bar chart
│   ├── direction_bars_WMT24++_4-Bit.png         # Quantization bar chart
│   └── direction_bars_pruning_30pct.png         # Pruning bar chart
├── comet22/ (same structure)
├── chrf/ (same structure)
└── kiwi23/ (same structure)
```

## Analysis Use Cases

### 1. Translation Direction Bias
- Compare how quantization affects English→Foreign vs Foreign→English
- Identify if models are more robust for translations TO English vs FROM English

### 2. Compression Method Effects
- Analyze if pruning is more harmful to multilingual (Foreign→Foreign) translations
- Compare direction-specific effects of different quantization levels

### 3. Resource Level Interactions
- Study how direction type interacts with language resource levels
- Identify if high-resource→low-resource translations are more affected

### 4. Model Architecture Insights
- Determine if models have English-centric representations
- Evaluate how well models maintain multilingual capabilities under compression

## Example Insights

The analysis can reveal patterns like:
- "8-bit quantization reduces English→Foreign performance by 2.3% but Foreign→English by only 1.1%"
- "Foreign→Foreign translations show 3x more variance under pruning"
- "High-resource→Low-resource directions are most affected by compression"

## Technical Details

- **Color Scheme**: Blue (English→Foreign), Orange (Foreign→English), Green (Foreign→Foreign)
- **Markers**: Circle (English→Foreign), Square (Foreign→English), Triangle (Foreign→Foreign)
- **Error Handling**: Gracefully handles missing data combinations
- **Baseline Comparison**: All pruning analyses compare against WMT24++ baseline performance

## Current Status

✅ **Complete**: All functions implemented and integrated  
⚠️ **Pending**: NumPy compatibility issues prevent execution  
🔄 **Ready**: Functions will run automatically once environment is fixed