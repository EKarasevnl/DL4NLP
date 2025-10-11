#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=wmt24pp_eval
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=15:00:00
#SBATCH --output=outputs/wmt24pp_eval_%A.out
#SBATCH --mem=128G

module purge
module load 2023
module load Anaconda3/2023.07-2

# Conda env
eval "$(/sw/arch/RHEL8/EB_production/2023/software/Anaconda3/2023.07-2/bin/conda shell.bash hook)"
conda activate nllb_final

cd /home/scur1832/DL4NLP
mkdir -p outputs/wmt24pp_eval

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Read HF token from secure file
if [ -f "/home/scur1832/DL4NLP/.hf_token" ]; then
    export HF_TOKEN=$(cat /home/scur1832/DL4NLP/.hf_token)
else
    echo "Warning: .hf_token file not found. Some models may not be accessible."
    export HF_TOKEN=""
fi

# Hugging Face login for accessing gated models like kiwi
# Option 1: Set HF_TOKEN environment variable before running sbatch
# Option 2: Login interactively (not recommended for batch jobs)
if [ -n "$HF_TOKEN" ]; then
    echo "Logging in to Hugging Face with provided token..."
    huggingface-cli login --token "$HF_TOKEN"
else
    echo "No HF_TOKEN provided. Some COMET models may not be accessible."
    echo "To use gated models, set HF_TOKEN environment variable:"
    echo "export HF_TOKEN=your_token_here"
    echo "sbatch jobs/run_final_test.sh"
fi

# 2-letter WMT codes for requested languages
LANGS=de,ru,fr,nl,pl,lv,zu,te,sw

# Allow overriding quantization mode and sparsity via env vars
if [ -z "$QUANTIZATION_MODE" ]; then
  QUANTIZATION_MODE="pruned"
  echo "QUANTIZATION_MODE not set, defaulting to $QUANTIZATION_MODE"
else
  echo "QUANTIZATION_MODE set to $QUANTIZATION_MODE"
fi

if [ -z "$SPARSITY" ]; then
  SPARSITY=0.1
  echo "SPARSITY not set, defaulting to $SPARSITY"
else
  echo "SPARSITY set to $SPARSITY"
fi

# Ensure local dataset presence (download once if missing)
DATA_DIR="/home/scur1832/DL4NLP/data/wmt24pp"
mkdir -p "$DATA_DIR"
if [ -z "$(ls -A "$DATA_DIR" 2>/dev/null)" ]; then
  echo "Local dataset not found; downloading google/wmt24pp to $DATA_DIR ..."
  huggingface-cli download --repo-type dataset google/wmt24pp --local-dir "$DATA_DIR" --local-dir-use-symlinks False | cat
fi

python wmt24pp_eval.py \
  --dataset "$DATA_DIR" \
  --langs "$LANGS" \
  --mode "$QUANTIZATION_MODE" \
  --sparsity "$SPARSITY" \
  --directions both \
  --split test \
  --batch_size 8 \
  --max_new_tokens 256 \
  --output_dir outputs/wmt24pp_eval \
  --sparsity 0.4

echo "Done. Scores written to outputs/wmt24pp_eval/scores.csv"



