export HF_TOKEN=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
export WANDB_DIR=/home/student/workspace/wandb
cd /home/student/workspace/skeleton/
source /home/student/workspace/skeleton/setup.sh

eval "$(conda shell.bash hook)"
conda activate $EVAL_ENV
python train_qwen.py "$@" 