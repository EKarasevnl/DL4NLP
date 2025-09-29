#!/bin/bash

# Batch submission script for parallel FLORES English evaluation
# This script submits separate jobs for each quantization mode (baseline, int8, int4)

echo "Submitting parallel FLORES English evaluation jobs..."
echo "Each job will evaluate English pairs using a different quantization mode"

# List of quantization modes
MODES=("baseline" "int8" "int4")

# Array to store job IDs
JOB_IDS=()

# Submit jobs for each quantization mode
for MODE in "${MODES[@]}"; do
    echo "Submitting FLORES job for quantization mode: $MODE"
    
    JOB_FILE="jobs/parallel_english_flores/run_flores_${MODE}.sh"
    
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
echo "Submitted ${#JOB_IDS[@]} parallel FLORES jobs"
echo "Job IDs: ${JOB_IDS[*]}"
echo ""
echo "Monitor progress with:"
echo "  squeue -u $USER"
echo ""
echo "Check outputs in:"
echo "  outputs/flores_eval/flores_*_*.out"
echo ""
echo "Results will be saved to:"
echo "  outputs/flores_eval/flores_scores_<mode>.csv"
echo ""
echo "To combine all results into one file after completion:"
echo "  head -1 outputs/flores_eval/flores_scores_baseline.csv > outputs/flores_eval/combined_flores_scores.csv"
echo "  tail -n +2 -q outputs/flores_eval/flores_scores_*.csv >> outputs/flores_eval/combined_flores_scores.csv"