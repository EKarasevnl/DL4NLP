#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=flores_eval
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=outputs/flores_eval_%A.out
#SBATCH --mem=128G

module purge
module load 2023
module load Anaconda3/2023.07-2

# Conda env
eval "$(/sw/arch/RHEL8/EB_production/2023/software/Anaconda3/2023.07-2/bin/conda shell.bash hook)"
conda activate nllb_final

cd /home/scur1844/DL4NLP
mkdir -p outputs/flores_eval

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

DATASET_DIR=/home/scur1844/.cache/huggingface/hub/datasets--facebook--flores/snapshots/2db78afdeaccaedc3b33a95442a4e55766887e17

# Languages from the screenshot
LANGS=deu_Latn,rus_Cyrl,fra_Latn,nld_Latn,pol_Latn,lvs_Latn,zul_Latn,tel_Telu,swh_Latn

echo "Starting FLORES eval for: $LANGS"

python flores_eval.py \
  --dataset_path "$DATASET_DIR" \
  --langs "$LANGS" \
  --mode int4 \
  --directions both \
  --split devtest \
  --batch_size 8 \
  --max_new_tokens 256 \
  --output_dir outputs/flores_eval

echo "Done. Scores written to outputs/flores_eval/scores.csv"



