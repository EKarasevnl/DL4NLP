#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=wmt24pp_int4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --output=outputs/wmt24pp_eval/wmt24pp_int4_%A.out
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
if [ -n "$HF_TOKEN" ]; then
    echo "Logging in to Hugging Face with provided token..."
    huggingface-cli login --token "$HF_TOKEN"
else
    echo "No HF_TOKEN provided. Some COMET models may not be accessible."
fi

echo "Starting WMT24++ eval for: deu_Latn,rus_Cyrl,fra_Latn,nld_Latn,pol_Latn,lvs_Latn,zul_Latn,tel_Telu,swh_Latn"
echo "Mode: int4"

python wmt24pp_eval_parallel.py   --dataset /home/scur1832/DL4NLP/data/wmt24pp   --langs deu_Latn,rus_Cyrl,fra_Latn,nld_Latn,pol_Latn,lvs_Latn,zul_Latn,tel_Telu,swh_Latn   --mode int4   --directions both   --batch_size 8   --max_new_tokens 256   --output_dir outputs/wmt24pp_eval

echo "Done. Scores written to outputs/wmt24pp_eval/wmt24pp_scores_int4.csv"
