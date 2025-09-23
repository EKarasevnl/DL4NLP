#!/bin/bash
#SBATCH --partition=gpu_mig
#SBATCH --gpus=1
#SBATCH --job-name=final_setup
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --output=outputs/final_setup_%A.out

module purge
module load 2023
module load Anaconda3/2023.07-2
# Use cluster CUDA only
module load CUDA/12.1.0

# Conda shell
eval "$(/sw/arch/RHEL8/EB_production/2023/software/Anaconda3/2023.07-2/bin/conda shell.bash hook)"

# Clean & create env
conda env remove -n nllb_final -y || true
conda create -y -n nllb_final python=3.10
conda activate nllb_final

# Build tools
conda install -y -c conda-forge cmake ninja packaging

# CUDA env from module
export CUDA_HOME=$EBROOTCUDA
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH}"

# PyTorch CUDA 12.1 (no torchvision/torchaudio needed for NLP)
pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121

# NLP stack
pip install "transformers==4.46.0" "accelerate==0.30.0" \
            "datasets==2.19.0" "sentencepiece==0.2.0" \
            "sacrebleu==2.4.0" "sacremoses==0.1.1" \
            "bitsandbytes==0.44.1" "huggingface_hub>=0.24.0" \
            "psutil" "tqdm"

# Dirs & caches
mkdir -p models/nllb_3.3b data/translation_datasets outputs logs hf_cache
export HF_HOME=$PWD/hf_cache
export TRANSFORMERS_CACHE=$PWD/hf_cache
export TOKENIZERS_PARALLELISM=false

# Quick smoke test (GPU + bnb)
python - <<'PY'
import torch, transformers, accelerate, bitsandbytes
print("GPU:", torch.cuda.is_available(), "Devices:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("Name:", torch.cuda.get_device_name(0))
print("Torch:", torch.__version__)
print("Transformers:", transformers.__version__)
print("Accelerate:", accelerate.__version__)
print("bitsandbytes:", bitsandbytes.__version__)
PY

echo "Environment ready for NLLB-200-3.3B quantized inference."
