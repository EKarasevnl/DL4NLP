# Parallel English Evaluation by Quantization Mode

This directory contains scripts for running English-centric evaluation (FLORES and WMT24++) in parallel, split by quantization mode rather than language pairs.

## Overview

Instead of running sequential evaluations with different quantization modes, this system allows:
- **Baseline (fp16/bf16)** - Full precision model
- **int8** - 8-bit quantized model  
- **int4** - 4-bit quantized model (NF4 + double quantization)

All running simultaneously on separate GPUs to compare quantization performance.

## Structure

- **Original approach**: Sequential evaluation with one mode at a time
- **Parallel approach**: 3 jobs for FLORES + 3 jobs for WMT24++ = 6 concurrent evaluations

Each job evaluates all English language pairs (en↔9 languages = 18 evaluations) using one quantization mode.

## Files Created

### New Evaluation Scripts (No Impact on Running Jobs)
- `flores_eval_parallel.py` - Modified FLORES script with mode-specific output filenames
- `wmt24pp_eval_parallel.py` - Modified WMT24++ script with mode-specific output filenames

### FLORES Parallel Jobs
- `jobs/parallel_english_flores/run_flores_baseline.sh` - Full precision FLORES evaluation
- `jobs/parallel_english_flores/run_flores_int8.sh` - 8-bit quantized FLORES evaluation  
- `jobs/parallel_english_flores/run_flores_int4.sh` - 4-bit quantized FLORES evaluation

### WMT24++ Parallel Jobs
- `jobs/parallel_english_wmt24pp/run_wmt24pp_baseline.sh` - Full precision WMT24++ evaluation
- `jobs/parallel_english_wmt24pp/run_wmt24pp_int8.sh` - 8-bit quantized WMT24++ evaluation
- `jobs/parallel_english_wmt24pp/run_wmt24pp_int4.sh` - 4-bit quantized WMT24++ evaluation

### Batch Submission Scripts
- `jobs/submit_parallel_flores.sh` - Submit all FLORES quantization jobs
- `jobs/submit_parallel_wmt24pp.sh` - Submit all WMT24++ quantization jobs  
- `jobs/submit_all_parallel_english.sh` - Submit both FLORES and WMT24++ jobs

## Usage

### Submit All English Parallel Jobs
```bash
cd /home/scur1832/DL4NLP
./jobs/submit_all_parallel_english.sh
```

### Submit Only FLORES Jobs
```bash
./jobs/submit_parallel_flores.sh
```

### Submit Only WMT24++ Jobs
```bash
./jobs/submit_parallel_wmt24pp.sh
```

## Output Files

### FLORES Results (by quantization mode)
- `outputs/flores_eval/flores_scores_baseline.csv`
- `outputs/flores_eval/flores_scores_int8.csv`
- `outputs/flores_eval/flores_scores_int4.csv`

### WMT24++ Results (by quantization mode)  
- `outputs/wmt24pp_eval/wmt24pp_scores_baseline.csv`
- `outputs/wmt24pp_eval/wmt24pp_scores_int8.csv`
- `outputs/wmt24pp_eval/wmt24pp_scores_int4.csv`

### Log Files
- `outputs/flores_eval/flores_<mode>_<jobid>.out`
- `outputs/wmt24pp_eval/wmt24pp_<mode>_<jobid>.out`

## Expected Performance Comparison

This parallel setup allows direct comparison of:
- **Baseline**: Highest quality, slowest inference, most memory
- **int8**: Good quality compromise, faster inference, less memory  
- **int4**: Fastest inference, least memory, potential quality trade-off

All evaluations use identical language pairs and datasets, enabling fair quantization analysis.

## Benefits

1. **Quantization Analysis** - Direct comparison of model compression effects
2. **Resource Efficiency** - Multiple quantization modes tested simultaneously
3. **Time Savings** - Parallel execution vs sequential quantization testing
4. **Isolated Testing** - New scripts don't interfere with currently running jobs
5. **Comprehensive Coverage** - Both FLORES-200 and WMT24++ datasets

## Compatibility

These new scripts are completely independent from the original evaluation scripts and currently running multilingual jobs. They use separate output directories and filenames to avoid conflicts.