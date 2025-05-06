import os

from transformers import set_seed
import json
import pandas as pd
import numpy as np

def load_data(base_dir):
    '''
    Load data from the specified directory and return a DataFrame.
    '''

    filenames = os.listdir(base_dir) # 파일명에 확장자 포함
    data_files = [os.path.join(base_dir, p) for p in filenames if ".json" in p]

    dataset = []
    for fn in data_files:
        with open(fn) as fp:
            data = json.load(fp)
        dataset.append(data)

    filenames = [fn.split(".")[0] for fn in filenames] # 확장자 제거
    data = []
    MAX_LEN = 1000 # train dataset의 최대 길이. 필요에 따라 조정 가능
    rng = np.random.default_rng(42)

    N = len(dataset)

    while len(data) < MAX_LEN:
        task_idx = rng.integers(0, N) # 랜덤으로 task 선택
        task = dataset[task_idx]
        file_name = filenames[task_idx]

        n_task = len(task)
        grids_idx =  rng.choice(n_task, size=4, replace=True) # 앞서 추출한 task에서 랜덤으로 4개의 grid 선택
        train_grids = [task[i] for i in grids_idx[:3]] # 3개는 train data로 사용
        test_grids = [task[i] for i in grids_idx[3:]] # 1개는 test data로 사용

        test_inputs = [{'input': grid['input']} for grid in test_grids]
        test_outputs = [grid['output'] for grid in test_grids]
        test_outputs_transformed = [{'output': grid} for grid in test_outputs]
        combined_tests = []
        for test_input, test_output in zip(test_inputs, test_outputs_transformed):
            combined_tests.append({'input': test_input['input'], 'output': test_output['output']})

        data.append({
            'task': file_name,
            'train': train_grids,
            'test_input': test_inputs,
            'test_output': test_outputs,
            'test': combined_tests,
        })

    df = pd.DataFrame(data)
    return df

def main():
    '''
    Main function to run the training process.
    '''
    token = os.environ.get("HF_TOKEN", None)
    from arc import ARCSolver

    solver = ARCSolver(token=token)

    set_seed(1234567890)

    data_path = "/workspace/dataset"
    val_size = 0.1

    df = load_data(data_path)

    from datasets import Dataset
    dataset = Dataset.from_pandas(df).shuffle(42)

    train_val_split = dataset.train_test_split(test_size=val_size, seed=42)
    train_dataset = train_val_split['train']
    val_dataset = train_val_split['test']

    solver.train(train_dataset, val_dataset)

if __name__ == "__main__":
    main()
