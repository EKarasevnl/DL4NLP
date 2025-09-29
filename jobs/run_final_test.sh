#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=final_test
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=outputs/final_test_%A.out
#SBATCH --mem=128G

module purge
module load 2023
module load Anaconda3/2023.07-2

# Conda env
eval "$(/sw/arch/RHEL8/EB_production/2023/software/Anaconda3/2023.07-2/bin/conda shell.bash hook)"
conda activate nllb_final
# pip install -U "bitsandbytes>=0.44.1"
# pip install unbabel-comet
# pip install evaluate
# pip install bert_score
# pip install "unbabel-comet>=2.1.0"
pip install "unbabel-comet>=2.1.0"

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

cd /home/scur1843/DL4NLP
# mkdir -p outputs

# Runtime environment (minimal & safe)
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=0   # 1 only for debugging slowdowns

# Job info
echo "=========================================="
echo "NLLB-200-3.3B Final Test Job"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Partition: $SLURM_JOB_PARTITION"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "=========================================="

# GPU check
python - <<'PY'
import torch
print(f"PyTorch: {torch.__version__}")
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device count:", torch.cuda.device_count())
    print("Current device:", torch.cuda.current_device())
    print("Device name:", torch.cuda.get_device_name(0))
    print("BF16 supported:", torch.cuda.is_bf16_supported())
PY

# Run translation tests (baseline, int8, int4)
python nllb_test.py --mode all --src eng_Latn --tgt spa_Latn --text "Hello from H100!"

echo "Test COMET installation:"

python -c "import comet; print('COMET available')"
python test_additional_metrics.py
