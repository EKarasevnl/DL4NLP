#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=multilang_zul
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --output=outputs/multilang_eval/multilang_zul_Latn_%A.out
#SBATCH --mem=128G

module purge
module load 2023
module load Anaconda3/2023.07-2

# Conda env
eval "$(/sw/arch/RHEL8/EB_production/2023/software/Anaconda3/2023.07-2/bin/conda shell.bash hook)"
conda activate nllb_final

cd /home/scur1832/DL4NLP
mkdir -p outputs/multilang_eval

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
if [ -n "$HF_TOKEN" ]; then
    echo "Logging in to Hugging Face with provided token..."
    huggingface-cli login --token "$HF_TOKEN"
else
    echo "No HF_TOKEN provided. Some COMET models may not be accessible."
    echo "To use gated models, set HF_TOKEN environment variable:"
    echo "export HF_TOKEN=your_token_here"
fi

# Dataset paths
FLORES_DIR="/home/scur1832/.cache/huggingface/modules/datasets_modules/datasets/facebook--flores"
WMT24PP_DIR="/home/scur1832/DL4NLP/data/wmt24pp"

# Languages (same as other evaluations)
LANGS=deu_Latn,rus_Cyrl,fra_Latn,nld_Latn,pol_Latn,lvs_Latn,zul_Latn,tel_Telu,swh_Latn

echo "Starting multilingual evaluation for source language: zul_Latn"
echo "Languages: $LANGS"
echo "FLORES path: $FLORES_DIR"
echo "WMT24++ path: $WMT24PP_DIR"

python multilang_eval.py   --flores_path "$FLORES_DIR"   --wmt24pp_path "$WMT24PP_DIR"   --langs "$LANGS"   --src_lang zul_Latn   --mode int4   --split devtest   --batch_size 8   --max_new_tokens 256   --output_dir outputs/multilang_eval

echo "Done. Scores written to outputs/multilang_eval/multilang_scores_zul_Latn.csv"
