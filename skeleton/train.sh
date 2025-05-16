export HF_TOKEN=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
cd /home/jupyter-b0neh3ad/dl-team14/skeleton
source ./setup.sh

eval "$(conda shell.bash hook)"
conda activate venv
python train.py --debug