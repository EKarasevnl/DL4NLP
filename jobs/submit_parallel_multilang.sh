#!/bin/bash

# Batch submission script for parallel multilingual evaluation
# This script submits separate jobs for each source language to speed up evaluation

echo "Submitting parallel multilingual evaluation jobs..."
echo "Each job will evaluate one source language against all other target languages"

# List of source languages
LANGUAGES=("deu_Latn" "rus_Cyrl" "fra_Latn" "nld_Latn" "pol_Latn" "lvs_Latn" "zul_Latn" "tel_Telu" "swh_Latn")

# Array to store job IDs
JOB_IDS=()

# Submit jobs for each source language
for SRC_LANG in "${LANGUAGES[@]}"; do
    echo "Submitting job for source language: $SRC_LANG"
    
    JOB_FILE="jobs/parallel_multilang/run_multilang_${SRC_LANG}.sh"
    
    # Submit job and capture job ID
    JOB_ID=$(sbatch --parsable "$JOB_FILE")
    JOB_IDS+=($JOB_ID)
    
    echo "  Job ID: $JOB_ID - $JOB_FILE"
    
    # Small delay to avoid overwhelming the scheduler
    sleep 1
done

echo ""
echo "Summary:"
echo "========="
echo "Submitted ${#JOB_IDS[@]} parallel jobs"
echo "Job IDs: ${JOB_IDS[*]}"
echo ""
echo "Monitor progress with:"
echo "  squeue -u $USER"
echo ""
echo "Check outputs in:"
echo "  outputs/multilang_eval/multilang_*_*.out"
echo ""
echo "Results will be saved to:"
echo "  outputs/multilang_eval/multilang_scores_<language>.csv"
echo ""
echo "To combine all results into one file after completion:"
echo "  cat outputs/multilang_eval/multilang_scores_*.csv | head -1 > outputs/multilang_eval/combined_multilang_scores.csv"
echo "  tail -n +2 -q outputs/multilang_eval/multilang_scores_*.csv >> outputs/multilang_eval/combined_multilang_scores.csv"