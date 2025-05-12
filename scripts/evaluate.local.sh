# clean previous eval directory
rm -rf /workspace/evaluate/*

# emulate submission process
# cp -r /home/student/workspace/skeleton/arc /workspace/evaluate
# cp -r /home/student/workspace/skeleton/artifacts /workspace/evaluate
# cp /home/student/workspace/skeleton/setup.sh /workspace/evaluate

cp -r /home/js1044k/dl-team14/skeleton/arc /workspace/evaluate
cp -r /home/js1044k/dl-team14/skeleton/artifacts /workspace/evaluate
cp /home/js1044k/dl-team14/skeleton/setup.sh /workspace/evaluate

cd /workspace/evaluate
source /workspace/evaluate/setup.sh
# cp /home/student/workspace/scripts/evaluate.py /workspace/evaluate

cp /home/js1044k/dl-team14/scripts/evaluate.py /workspace/evaluate
conda run -n $EVAL_ENV python evaluate.py
