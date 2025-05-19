# clean previous eval directory
rm -rf /workspace/evaluate/*

# emulate submission process
# cp -r /home/student/workspace/skeleton/arc /workspace/evaluate
# cp -r /home/student/workspace/skeleton/artifacts /workspace/evaluate
# cp /home/student/workspace/skeleton/setup.sh /workspace/evaluate

cp -r ~/dl-team14/skeleton/arc /workspace/evaluate
cp -r ~/dl-team14/skeleton/artifacts /workspace/evaluate
cp ~/dl-team14/skeleton/setup.sh /workspace/evaluate

cd /workspace/evaluate
source /workspace/evaluate/setup.sh
cp ~/dl-team14/scripts/evaluate.py /workspace/evaluate

eval "$(conda shell.bash hook)"
conda activate $EVAL_ENV
python evaluate.py
