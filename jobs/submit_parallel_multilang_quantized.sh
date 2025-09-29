#!/bin/bash

# Batch submission script for parallel multilingual evaluation by quantization mode
# This script submits separate jobs for each source language and quantization mode combination

echo "Submitting parallel multilingual evaluation jobs by quantization mode..."
echo "Each job evaluates one source language with one quantization mode"

# List of source languages and modes
LANGUAGES=("deu_Latn" "rus_Cyrl" "fra_Latn" "nld_Latn" "pol_Latn" "lvs_Latn" "zul_Latn" "tel_Telu" "swh_Latn")
MODES=("baseline" "int8" "int4")

# Array to store job IDs
JOB_IDS=()

# Submit jobs for each source language and mode combination
for SRC_LANG in "${LANGUAGES[@]}"; do
    for MODE in "${MODES[@]}"; do
        echo "Submitting multilingual job for language: $SRC_LANG, mode: $MODE"
        
        JOB_FILE="jobs/parallel_multilang_quantized/run_multilang_${SRC_LANG}_${MODE}.sh"
        
        # Submit job and capture job ID
        JOB_ID=$(sbatch --parsable "$JOB_FILE")
        JOB_IDS+=($JOB_ID)
        
        echo "  Job ID: $JOB_ID - $JOB_FILE"
        
        # Small delay to avoid overwhelming the scheduler
        sleep 0.5
    done
done

echo ""
echo "Summary:"
echo "========="
echo "Submitted ${#JOB_IDS[@]} parallel multilingual jobs"
echo "Structure: 9 languages × 3 modes = ${#JOB_IDS[@]} jobs"
echo "Job IDs: ${JOB_IDS[*]}"
echo ""
echo "Monitor progress with:"
echo "  squeue -u $USER"
echo ""
echo "Check outputs in:"
echo "  outputs/multilang_eval/multilang_*_*_*.out"
echo ""
echo "Results will be saved to:"
echo "  outputs/multilang_eval/multilang_scores_<language>_<mode>.csv"
echo ""
echo "To combine results by mode after completion:"
echo "  for mode in baseline int8 int4; do"
echo "    head -1 outputs/multilang_eval/multilang_scores_deu_Latn_\$mode.csv > outputs/multilang_eval/combined_multilang_\$mode.csv"
echo "    tail -n +2 -q outputs/multilang_eval/multilang_scores_*_\$mode.csv >> outputs/multilang_eval/combined_multilang_\$mode.csv"
echo "  done"