# NLLB-200-3.3B Translation Evaluation Suite

Evaluation scripts for the NLLB-200-3.3B translation model with support for quantization and pruning methods.

## Python Scripts

### eval_utils.py
Shared utilities module containing model loading functions (baseline/int8/int4/pruned), batch translation, metrics computation (BLEU, chrF, COMET-22, kiwi-23), and language code mappings.

### wmt24pp_eval.py
Evaluates translation quality on WMT24++ dataset for English ↔ target language pairs. Supports different quantization modes and both translation directions.

### wmt24pp_multilang_eval.py
Evaluates direct translation between non-English language pairs using the WMT24++ dataset with English pivoting.

### flores_eval_parallel.py
Runs translation evaluation on the FLORES-200 benchmark dataset for English ↔ target language pairs.

### multilang_eval.py
Comprehensive evaluation across all non-English language pairs using both FLORES-200 and WMT24++ datasets.

## Usage

All scripts support different model modes:
- `baseline` - Full precision (fp16/bf16)
- `int8` - 8-bit quantization
- `int4` - 4-bit quantization  
- `pruned` - Magnitude pruning (requires `--sparsity` parameter)

Basic example:
```bash
python wmt24pp_eval.py --langs de,ru,fr --mode int4 --directions both
```

## Metrics

All scripts compute: BLEU, chrF, COMET-22 (reference-based), and kiwi-23 (reference-free).

## Output

Results are saved as CSV files in the specified output directory with columns: language pair, mode, and metric scores.

## Collaborators

- **Egor Karasev**
- **Iwo Godzwon**
- **Pradyut Nair**
- **Jan Henrik Bertrand**
- **Louis Gehringer**

---
