#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=flores_int8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --output=outputs/flores_eval/flores_int8_%A.out
#SBATCH --mem=128G

module purge
module load 2023
module load Anaconda3/2023.07-2

# Conda env
eval "$(/sw/arch/RHEL8/EB_production/2023/software/Anaconda3/2023.07-2/bin/conda shell.bash hook)"
conda activate nllb_final

cd /home/scur1832/DL4NLP
mkdir -p outputs/flores_eval

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
fi

DATASET_DIR="/home/scur1832/.cache/huggingface/modules/datasets_modules/datasets/facebook--flores/2a1174c8c4991ca09a9cb5b9a367cb2e049b073852cb4097456164d4612391ef/"

echo "Starting FLORES eval for: deu_Latn,rus_Cyrl,fra_Latn,nld_Latn,pol_Latn,lvs_Latn,zul_Latn,tel_Telu,swh_Latn"
echo "Mode: int8"

python flores_eval_parallel.py   --dataset_path "$DATASET_DIR"   --langs deu_Latn,rus_Cyrl,fra_Latn,nld_Latn,pol_Latn,lvs_Latn,zul_Latn,tel_Telu,swh_Latn   --mode int8   --directions both   --split devtest   --batch_size 8   --max_new_tokens 256   --output_dir outputs/flores_eval

echo "Done. Scores written to outputs/flores_eval/flores_scores_int8.csv"
