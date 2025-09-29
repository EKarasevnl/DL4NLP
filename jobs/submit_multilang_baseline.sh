#!/bin/bash

# Submit only baseline multilingual jobs
echo "Submitting baseline multilingual evaluation jobs..."

LANGUAGES=("deu_Latn" "rus_Cyrl" "fra_Latn" "nld_Latn" "pol_Latn" "lvs_Latn" "zul_Latn" "tel_Telu" "swh_Latn")
JOB_IDS=()

for SRC_LANG in "${LANGUAGES[@]}"; do
    echo "Submitting baseline job for language: $SRC_LANG"
    JOB_FILE="jobs/parallel_multilang_quantized/run_multilang_${SRC_LANG}_baseline.sh"
    JOB_ID=$(sbatch --parsable "$JOB_FILE")
    JOB_IDS+=($JOB_ID)
    echo "  Job ID: $JOB_ID"
    sleep 0.5
done

echo "Submitted ${#JOB_IDS[@]} baseline multilingual jobs"
echo "Job IDs: ${JOB_IDS[*]}"