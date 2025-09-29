#!/bin/bash

# Submit only int8 multilingual jobs
echo "Submitting int8 multilingual evaluation jobs..."

LANGUAGES=("deu_Latn" "rus_Cyrl" "fra_Latn" "nld_Latn" "pol_Latn" "lvs_Latn" "zul_Latn" "tel_Telu" "swh_Latn")
JOB_IDS=()

for SRC_LANG in "${LANGUAGES[@]}"; do
    echo "Submitting int8 job for language: $SRC_LANG"
    JOB_FILE="jobs/parallel_multilang_quantized/run_multilang_${SRC_LANG}_int8.sh"
    JOB_ID=$(sbatch --parsable "$JOB_FILE")
    JOB_IDS+=($JOB_ID)
    echo "  Job ID: $JOB_ID"
    sleep 0.5
done

echo "Submitted ${#JOB_IDS[@]} int8 multilingual jobs"
echo "Job IDs: ${JOB_IDS[*]}"