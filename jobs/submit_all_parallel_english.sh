#!/bin/bash

# Batch submission script for all parallel English evaluation jobs
# This script submits both FLORES and WMT24++ jobs for all quantization modes

echo "Submitting all parallel English evaluation jobs..."
echo "This includes both FLORES and WMT24++ evaluations across all quantization modes"

# Submit FLORES parallel jobs
echo "=== FLORES PARALLEL JOBS ==="
./jobs/submit_parallel_flores.sh

echo ""
echo "=== WMT24++ PARALLEL JOBS ==="
./jobs/submit_parallel_wmt24pp.sh

echo ""
echo "All English evaluation jobs submitted!"
echo "Total jobs: 6 (3 FLORES + 3 WMT24++)"
echo ""
echo "Monitor all jobs with:"
echo "  squeue -u $USER"