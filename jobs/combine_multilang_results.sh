#!/bin/bash

# Script to combine parallel multilingual evaluation results
# Run this after all parallel jobs have completed

OUTPUT_DIR="outputs/multilang_eval"
COMBINED_FILE="$OUTPUT_DIR/combined_multilang_scores.csv"

echo "Combining parallel multilingual evaluation results..."

# Check if individual result files exist
RESULT_FILES=("$OUTPUT_DIR"/multilang_scores_*.csv)

if [ ! -e "${RESULT_FILES[0]}" ]; then
    echo "Error: No result files found in $OUTPUT_DIR"
    echo "Expected files: multilang_scores_*.csv"
    exit 1
fi

echo "Found ${#RESULT_FILES[@]} result files:"
for file in "${RESULT_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "  $file"
    fi
done

# Create combined file with header
echo "Creating combined results file: $COMBINED_FILE"

# Get header from first file
head -1 "${RESULT_FILES[0]}" > "$COMBINED_FILE"

# Append all data (excluding headers)
for file in "${RESULT_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "Processing: $(basename "$file")"
        tail -n +2 "$file" >> "$COMBINED_FILE"
    fi
done

# Count total evaluations
TOTAL_LINES=$(wc -l < "$COMBINED_FILE")
TOTAL_EVALUATIONS=$((TOTAL_LINES - 1))  # Subtract header

echo ""
echo "Summary:"
echo "========"
echo "Combined file: $COMBINED_FILE"
echo "Total evaluations: $TOTAL_EVALUATIONS"
echo "Expected evaluations: 72 (9 languages × 8 targets each)"

# Verify we have the expected number of evaluations
if [ "$TOTAL_EVALUATIONS" -eq 72 ]; then
    echo "✅ All evaluations completed successfully!"
else
    echo "⚠️  Warning: Expected 72 evaluations, found $TOTAL_EVALUATIONS"
fi

echo ""
echo "Individual result files are preserved in: $OUTPUT_DIR"
echo "Combined results available in: $COMBINED_FILE"