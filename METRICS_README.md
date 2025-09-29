# Additional Metrics Documentation

## Overview

Both `flores_eval.py` and `wmt24pp_eval.py` have been enhanced to include three additional evaluation metrics alongside the existing BLEU and chrF scores:

### 1. COMET-22
- **Description**: Reference-based neural metric using the WMT22 COMET-DA model
- **Range**: Typically 0-1 (higher is better)
- **Package**: `unbabel-comet`
- **Model**: `Unbabel/wmt22-comet-da`
- **Use case**: Provides neural evaluation that correlates better with human judgments than traditional metrics

### 2. kiwi-23  
- **Description**: Reference-free quality estimation metric using COMET-kiwi (WMT22 model)
- **Range**: Typically 0-1 (higher is better)
- **Package**: `unbabel-comet`
- **Model**: `Unbabel/wmt22-cometkiwi-da` (reference-free quality estimation)
- **Use case**: Quality estimation without reference translations (useful for production scenarios)

### 3. XL (Cross-lingual)
- **Description**: Cross-lingual semantic similarity using XLM-RoBERTa-based BERTScore
- **Range**: Typically 0-1 (higher is better)
- **Package**: `evaluate` + `bert-score`
- **Model**: `xlm-roberta-large`
- **Use case**: Measures semantic similarity across languages using multilingual embeddings

## Installation

Install the required dependencies:

```bash
pip install -r requirements_additional_metrics.txt
```

## Usage

The evaluation scripts will automatically compute these metrics if the required packages are available. If a package is missing, the script will show a warning and continue with available metrics.

### Output Format

The CSV output now includes additional columns:
- `lang`: Language code
- `direction`: Translation direction (en2x, x2en)
- `mode`: Model quantization mode (baseline, int8, int4)
- `split`: Dataset split used
- `bleu`: BLEU score
- `chrf`: chrF score
- `comet22`: COMET-22 score (or "N/A" if unavailable)
- `kiwi23`: kiwi-23 score (or "N/A" if unavailable)
- `xl`: XL metric score (or "N/A" if unavailable)
- `num_sentences`: Number of sentences evaluated

### Console Output

The console output now displays all available metrics:
```
BLEU: 25.4  chrF: 52.1  COMET-22: 0.8234  kiwi-23: 0.7891  XL: 0.8456  N: 1012  time: 45.2s
```

## Error Handling

- If any metric fails to compute, it will show a warning but continue with other metrics
- Missing dependencies are handled gracefully with informative messages
- Individual metric failures don't stop the overall evaluation process

## Performance Considerations

- COMET metrics require GPU for optimal performance
- XL metric computation can be memory-intensive for large datasets
- Consider reducing batch size if encountering memory issues
- The additional metrics will increase evaluation time significantly

## Testing

Run the test script to verify the implementation:

```bash
python test_additional_metrics.py
```