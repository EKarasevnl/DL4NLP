# NLLB-200-3.3B Translation Evaluation Suite

This repository contains evaluation scripts for the NLLB-200-3.3B (No Language Left Behind) translation model across multiple datasets and language pairs, with support for various quantization and pruning methods.

## Repository Structure

```
DL4NLP/
├── eval_utils.py                 # Shared utility functions for all evaluation scripts
├── wmt24pp_eval.py              # WMT24++ dataset evaluation (English <-> X)
├── wmt24pp_multilang_eval.py    # WMT24++ non-English language pairs (X <-> Y)
├── flores_eval_parallel.py      # FLORES-200 dataset evaluation
├── multilang_eval.py            # Multi-language pair evaluation (FLORES + WMT24++)
├── data/                        # Dataset files
├── outputs/                     # Evaluation results (CSV files)
├── logs/                        # Job logs
└── jobs/                        # SLURM job scripts
    ├── run_*.sh                 # Individual evaluation job scripts
    ├── submit_*.sh              # Batch submission scripts  
    ├── combine_*.sh             # Result aggregation scripts
    └── parallel_*/              # Parallel execution scripts by dataset/mode
```

## Core Components

### eval_utils.py - Shared Utilities

Central module containing shared functionality used by all evaluation scripts:

**Environment Utilities:**
- `supports_bf16()` - Check if bfloat16 is supported
- `get_dtype_baseline()` - Get appropriate dtype for baseline model
- `print_env()` - Print system environment information

**Model Loading:**
- `load_model_baseline(model_id, dtype)` - Load model in fp16/bf16
- `load_model_int8(model_id)` - Load model with LLM.int8 quantization
- `load_model_int4(model_id)` - Load model with NF4 quantization + double quantization
- `load_model_pruned(model_id, sparsity)` - Load model with magnitude pruning
- `prepare_model_and_tokenizer(model_id, mode, sparsity)` - Unified model/tokenizer loader

**Translation:**
- `batch_translate(model, tokenizer, inputs_texts, src_lang, tgt_lang, batch_size, max_new_tokens)` - Batch translation with progress tracking

**Metrics:**
- `compute_advanced_metrics(sources, hypotheses, references)` - Compute COMET-22 and kiwi-23 scores

**Language Mappings:**
- `NLLB_TO_WMT24PP` - Map NLLB codes to WMT24++ codes
- `WMT24PP_TO_NLLB` - Map WMT24++ codes to NLLB codes

**Helpers:**
- `get_scores_filename(mode, sparsity)` - Generate standardized output filenames
- `format_metric(value)` - Format metric values for display

## Evaluation Scripts

### 1. wmt24pp_eval.py - WMT24++ English Pairs

Evaluates translations between English and target languages using the WMT24++ dataset.

**Usage:**
```bash
python wmt24pp_eval.py \
    --dataset wmt24pp-dataset \
    --langs de,ru,fr,nl,pl,lv,zu,te,sw \
    --directions both \
    --mode int4 \
    --batch_size 8 \
    --max_new_tokens 256 \
    --output_dir outputs/wmt24pp_eval
```

**Supported Modes:**
- `baseline` - Full precision (fp16/bf16)
- `int8` - LLM.int8 quantization
- `int4` - NF4 quantization with double quantization
- `pruned` - Magnitude pruning (requires `--sparsity` parameter)

**Directions:**
- `en2x` - English to target language
- `x2en` - Target language to English
- `both` - Both directions

**Output:** CSV file with BLEU, chrF, COMET-22, and kiwi-23 scores per language pair.

### 2. wmt24pp_multilang_eval.py - WMT24++ Non-English Pairs

Evaluates direct translation between non-English language pairs using WMT24++ dataset with English pivoting.

**Usage:**
```bash
python wmt24pp_multilang_eval.py \
    --wmt24pp_path data/wmt24pp \
    --source_lang deu_Latn \
    --target_langs rus_Cyrl,fra_Latn,nld_Latn,pol_Latn \
    --mode int4 \
    --batch_size 8 \
    --max_new_tokens 256 \
    --max_examples 1000 \
    --output_dir outputs/wmt24pp_multilang
```

**Parameters:**
- `--source_lang` - Source language (NLLB code)
- `--target_langs` - Comma-separated list of target languages
- `--max_examples` - Maximum number of sentence pairs to evaluate

**Output:** CSV file with evaluation results per language pair.

### 3. flores_eval_parallel.py - FLORES-200 Evaluation

Evaluates translation quality on the FLORES-200 benchmark dataset.

**Usage:**
```bash
python flores_eval_parallel.py \
    --dataset_path /path/to/flores/snapshot \
    --langs deu_Latn,rus_Cyrl,fra_Latn,nld_Latn,pol_Latn,lvs_Latn,zul_Latn,tel_Telu,swh_Latn \
    --directions both \
    --mode int4 \
    --batch_size 8 \
    --max_new_tokens 256 \
    --output_dir outputs/flores_eval
```

**Features:**
- Supports local FLORES dataset snapshots
- Evaluates English <-> target language pairs
- Parallel processing of language pairs

**Output:** CSV file with FLORES devtest results.

### 4. multilang_eval.py - Multi-Language Evaluation

Comprehensive evaluation across all non-English language pairs using both FLORES and WMT24++ datasets.

**Usage:**
```bash
python multilang_eval.py \
    --flores_path /path/to/flores \
    --wmt24pp_path data/wmt24pp \
    --langs deu_Latn,rus_Cyrl,fra_Latn,nld_Latn,pol_Latn,lvs_Latn,zul_Latn,tel_Telu,swh_Latn \
    --mode int4 \
    --batch_size 8 \
    --max_new_tokens 256 \
    --output_dir outputs/multilang_eval
```

**Modes:**
- **All pairs** (default): Evaluate all combinations of language pairs
- **Parallel mode**: Evaluate only from a specific source language using `--src_lang`

**Output:** CSV file with results for all evaluated pairs.

## Supported Languages

The following 9 languages are evaluated (NLLB language codes):

| Language | NLLB Code | WMT24++ Code |
|----------|-----------|--------------|
| German | `deu_Latn` | `de` |
| Russian | `rus_Cyrl` | `ru` |
| French | `fra_Latn` | `fr` |
| Dutch | `nld_Latn` | `nl` |
| Polish | `pol_Latn` | `pl` |
| Latvian | `lvs_Latn` | `lv` |
| Zulu | `zul_Latn` | `zu` |
| Telugu | `tel_Telu` | `te` |
| Swahili | `swh_Latn` | `sw` |

## Evaluation Metrics

All scripts compute the following metrics:

1. **BLEU** - Bilingual Evaluation Understudy (corpus-level)
2. **chrF** - Character n-gram F-score
3. **COMET-22** - Reference-based quality estimation (Unbabel/wmt22-comet-da)
4. **kiwi-23** - Reference-free quality estimation (Unbabel/wmt23-cometkiwi-da-xxl)

## Quantization & Optimization Methods

### Baseline (fp16/bf16)
Full precision model using float16 or bfloat16 (if supported by hardware).

```bash
--mode baseline
```

### INT8 Quantization (LLM.int8)
8-bit integer quantization using the LLM.int8 method from bitsandbytes.

```bash
--mode int8
```

### INT4 Quantization (NF4)
4-bit NormalFloat quantization with double quantization for reduced memory footprint.

```bash
--mode int4
```

### Magnitude Pruning
Structured pruning based on weight magnitudes. Requires specifying sparsity level.

```bash
--mode pruned --sparsity 0.5
```

Sparsity values:
- `0.3` = 30% of weights pruned
- `0.5` = 50% of weights pruned (default)
- `0.7` = 70% of weights pruned

## Running Evaluations

### Quick Start

1. **Install dependencies:**
```bash
pip install torch transformers datasets sacrebleu unbabel-comet bitsandbytes accelerate
```

2. **Run a basic evaluation:**
```bash
python wmt24pp_eval.py \
    --dataset wmt24pp-dataset \
    --langs de,ru,fr \
    --mode int4 \
    --directions both \
    --output_dir outputs/test
```

3. **Check results:**
```bash
cat outputs/test/scores_int4.csv
```

### SLURM Cluster Execution

For HPC environments, use the job scripts in the `jobs/` directory.

#### Main Job Scripts

**Individual Evaluation Jobs:**
- `jobs/run_wmt24pp_eval.sh` - WMT24++ English pairs evaluation
- `jobs/run_wmt24pp_multilang.sh` - WMT24++ multilingual evaluation
- `jobs/run_multilang_eval.sh` - Full multilingual evaluation (FLORES + WMT24++)
- `jobs/run_wmt24pp_multilang_<lang>.sh` - Per-language WMT24++ multilingual jobs

**Batch Submission Scripts:**
- `jobs/submit_parallel_wmt24pp.sh` - Submit parallel WMT24++ English pairs jobs
- `jobs/submit_parallel_flores.sh` - Submit parallel FLORES evaluation jobs
- `jobs/submit_parallel_multilang.sh` - Submit parallel multilingual jobs (all pairs)
- `jobs/submit_parallel_multilang_quantized.sh` - Submit multilingual jobs with quantization
- `jobs/submit_all_wmt24pp_multilang.sh` - Submit all WMT24++ multilingual jobs
- `jobs/submit_all_parallel_english.sh` - Submit all English pair evaluations

**Result Aggregation:**
- `jobs/combine_multilang_results.sh` - Combine results from parallel multilingual evaluations
- `jobs/combine_wmt24_multilang_results.sh` - Combine WMT24++ multilingual results

#### Example: Submit Parallel WMT24++ Evaluation

```bash
# Submit jobs for all quantization modes
cd jobs
bash submit_parallel_wmt24pp.sh

# Check job status
squeue -u $USER

# Once complete, check results
ls -lh ../outputs/wmt24pp_eval/
```

#### Example: Submit Multilingual Evaluation

```bash
# Submit parallel multilingual jobs for int4 mode
cd jobs
bash submit_parallel_multilang_quantized.sh

# Monitor progress
watch -n 60 'squeue -u $USER | grep multilang'

# Combine results when done
bash combine_multilang_results.sh
```

#### Parallel Execution Directories

The `jobs/` directory contains subdirectories for organized parallel execution:

- `parallel_english_flores/` - FLORES English pairs (baseline, int4, int8)
- `parallel_english_wmt24pp/` - WMT24++ English pairs (baseline, int4, int8)
- `parallel_multilang/` - Multilingual pairs per source language
- `parallel_multilang_quantized/` - Multilingual with different quantization modes

Each contains individual job scripts for specific language/mode combinations.

## Output Format

All evaluation scripts produce CSV files with the following columns:

| Column | Description |
|--------|-------------|
| `lang` / `src_lang` | Source or target language code |
| `direction` / `tgt_lang` | Translation direction or target language |
| `mode` | Quantization/optimization mode |
| `split` | Dataset split (e.g., "devtest") |
| `bleu` | BLEU score |
| `chrf` | chrF score |
| `comet22` | COMET-22 score (reference-based) |
| `kiwi23` | kiwi-23 score (reference-free) |
| `num_sentences` | Number of sentences evaluated |

For pruned models, filenames include sparsity: `scores_pruned_s0p50.csv`

## Example Workflows

### 1. Compare Quantization Methods

Evaluate the same language pairs with different quantization modes:

```bash
# Baseline
python wmt24pp_eval.py --langs de,fr,ru --mode baseline --directions both --output_dir outputs/comparison

# INT8
python wmt24pp_eval.py --langs de,fr,ru --mode int8 --directions both --output_dir outputs/comparison

# INT4
python wmt24pp_eval.py --langs de,fr,ru --mode int4 --directions both --output_dir outputs/comparison

# Compare results
paste outputs/comparison/scores_baseline.csv outputs/comparison/scores_int8.csv outputs/comparison/scores_int4.csv
```

### 2. Evaluate Pruning at Different Sparsity Levels

```bash
for sparsity in 0.3 0.5 0.7; do
    python wmt24pp_eval.py \
        --langs de,fr,ru \
        --mode pruned \
        --sparsity $sparsity \
        --directions both \
        --output_dir outputs/pruning
done

# Results in: scores_pruned_s0p30.csv, scores_pruned_s0p50.csv, scores_pruned_s0p70.csv
```

### 3. Full Multi-Language Matrix

Evaluate all language pairs:

```bash
python multilang_eval.py \
    --flores_path /path/to/flores \
    --wmt24pp_path data/wmt24pp \
    --langs deu_Latn,rus_Cyrl,fra_Latn,nld_Latn,pol_Latn,lvs_Latn,zul_Latn,tel_Telu,swh_Latn \
    --mode int4 \
    --output_dir outputs/full_matrix
```

This evaluates all 72 language pairs (9×8 combinations).

## Configuration

### Common Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--model` | HuggingFace model ID | `facebook/nllb-200-3.3B` |
| `--batch_size` | Translation batch size | `8` |
| `--max_new_tokens` | Maximum tokens to generate | `256` |
| `--output_dir` | Output directory for results | `outputs/` |

### GPU Memory Requirements

Approximate VRAM usage per mode:

- **Baseline (fp16/bf16)**: ~13 GB
- **INT8**: ~7 GB
- **INT4**: ~4 GB
- **Pruned (50%)**: ~6-7 GB

## Troubleshooting

### Issue: COMET model fails to load

**Solution:** Ensure unbabel-comet is installed:
```bash
pip install unbabel-comet
```

### Issue: CUDA out of memory

**Solution:** Reduce batch size or use more aggressive quantization:
```bash
--batch_size 4 --mode int4
```

### Issue: Dataset not found

**Solution:** For FLORES, provide the full path to the dataset snapshot directory:
```bash
--dataset_path /home/user/.cache/huggingface/hub/datasets--facebook--flores/snapshots/<hash>
```

### Issue: Job fails on SLURM

**Solution:** Check the output log files in `outputs/` directory:
```bash
# Find recent output files
ls -lt outputs/*.out | head -5

# Check for errors
grep -i error outputs/your_job_12345.out
```

## References

- **NLLB-200**: [No Language Left Behind](https://ai.facebook.com/research/no-language-left-behind/)
- **FLORES-200**: [FLORES Benchmark](https://github.com/facebookresearch/flores)
- **WMT24++**: Extended WMT evaluation dataset
- **COMET**: [Crosslingual Optimized Metric for Evaluation of Translation](https://github.com/Unbabel/COMET)
- **bitsandbytes**: [8-bit & 4-bit Quantization](https://github.com/TimDettmers/bitsandbytes)

## License

See LICENSE file for details.

## Contributing

This repository is organized for clarity and maintainability:

- All shared code is in `eval_utils.py`
- Each evaluation script focuses on a specific task
- No code duplication across files
- Consistent output format across all scripts
- Job scripts organized by evaluation type and execution mode

When adding new evaluation scripts, import shared functionality from `eval_utils.py` rather than duplicating code.

## Repository Maintenance

This repository has been refactored for optimal organization:

- **Code Deduplication** - All shared utilities centralized in `eval_utils.py`  
- **Clean Structure** - Only essential evaluation scripts and job scripts retained  
- **Consistent API** - Uniform function signatures across all modules  
- **Comprehensive Documentation** - Complete usage examples and API reference

---

**Last Updated:** 2025-10-11
