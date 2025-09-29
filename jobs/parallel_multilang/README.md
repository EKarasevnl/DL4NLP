# Parallel Multilingual Evaluation

This directory contains scripts for running multilingual evaluation in parallel to speed up the overall process.

## Overview

The original `multilang_eval.py` evaluates all non-English language pairs sequentially, which takes about 2.6 hours for 72 evaluations. The parallel system splits this work by source language, allowing multiple jobs to run simultaneously.

## Structure

- **Original approach**: 1 job × 72 evaluations = ~2.6 hours
- **Parallel approach**: 9 jobs × 8 evaluations each = ~20 minutes (estimated)

Each job evaluates one source language against all other target languages (8 evaluations per job).

## Files

### Core Scripts
- `multilang_eval.py` - Modified to accept `--src_lang` parameter for parallel processing
- `jobs/parallel_multilang/run_multilang_<language>.sh` - Individual job scripts (9 files)
- `jobs/submit_parallel_multilang.sh` - Batch submission script
- `jobs/combine_multilang_results.sh` - Results combination script

### Generated Jobs
- `run_multilang_deu_Latn.sh` - German → all other languages
- `run_multilang_rus_Cyrl.sh` - Russian → all other languages  
- `run_multilang_fra_Latn.sh` - French → all other languages
- `run_multilang_nld_Latn.sh` - Dutch → all other languages
- `run_multilang_pol_Latn.sh` - Polish → all other languages
- `run_multilang_lvs_Latn.sh` - Latvian → all other languages
- `run_multilang_zul_Latn.sh` - Zulu → all other languages
- `run_multilang_tel_Telu.sh` - Telugu → all other languages
- `run_multilang_swh_Latn.sh` - Swahili → all other languages

## Usage

### 1. Submit All Parallel Jobs
```bash
cd /home/scur1832/DL4NLP
./jobs/submit_parallel_multilang.sh
```

This will:
- Submit 9 separate jobs to the SLURM scheduler
- Each job runs on one H100 GPU for up to 8 hours
- Jobs can run simultaneously if resources are available

### 2. Monitor Progress
```bash
# Check job status
squeue -u scur1832

# Check individual outputs
tail -f outputs/multilang_eval/multilang_deu_Latn_<jobid>.out
```

### 3. Combine Results
After all jobs complete:
```bash
./jobs/combine_multilang_results.sh
```

This creates `outputs/multilang_eval/combined_multilang_scores.csv` with all results.

## Output Files

### Individual Results
- `outputs/multilang_eval/multilang_scores_deu_Latn.csv`
- `outputs/multilang_eval/multilang_scores_rus_Cyrl.csv`
- `outputs/multilang_eval/multilang_scores_fra_Latn.csv`
- etc. (one per source language)

### Combined Results
- `outputs/multilang_eval/combined_multilang_scores.csv` - All results in one file

### Log Files
- `outputs/multilang_eval/multilang_<language>_<jobid>.out` - Job execution logs

## Expected Performance

- **Time per job**: ~20 minutes (8 evaluations per job)
- **Total wall time**: ~20 minutes (if all jobs run in parallel)
- **Total evaluations**: 72 (same as original)
- **Resource usage**: 9 H100 GPUs simultaneously vs 1 H100 GPU sequentially

## Troubleshooting

### If jobs fail to start
- Check SLURM queue: `squeue -u scur1832`
- Check resource availability: `sinfo -p gpu_h100`

### If individual jobs fail
- Check specific job logs in `outputs/multilang_eval/`
- Re-run individual jobs manually:
  ```bash
  sbatch jobs/parallel_multilang/run_multilang_<language>.sh
  ```

### Missing results
- Ensure all jobs completed successfully before combining
- Check that 9 result files exist in `outputs/multilang_eval/`
- Expected: 72 total evaluations across all files

## Compatibility

This parallel system produces identical results to the original sequential approach, just much faster. The `multilang_eval.py` script supports both modes:

- **Parallel mode**: `--src_lang <language>` (used by job scripts)
- **Sequential mode**: No `--src_lang` parameter (original behavior)