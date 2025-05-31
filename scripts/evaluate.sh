# clean previous eval directory
rm -rf /workspace/evaluate/*

# emulate submission process
cp -r /home/student/workspace/1st-sol/submit/arc /workspace/evaluate
cp -r /home/student/workspace/1st-sol/submit/artifacts /workspace/evaluate
cp /home/student/workspace/1st-sol/submit/setup.sh /workspace/evaluate

cd /workspace/evaluate
source /workspace/evaluate/setup.sh
cp /home/student/workspace/scripts/evaluate.py /workspace/evaluate

eval "$(conda shell.bash hook)"
conda activate $EVAL_ENV
python evaluate.py
