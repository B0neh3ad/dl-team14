export HF_TOKEN=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
cd ~/dl-team14/skeleton/
source setup.sh

eval "$(conda shell.bash hook)"
conda activate $EVAL_ENV
python train.py --lora-r 128 --learning-rate 5e-5 --train-batch-size 4 --grad-acc-steps 4