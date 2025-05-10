import os

from transformers import set_seed
import json
import pandas as pd
import numpy as np
import wandb

DEBUG = True

class cfg:
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    output_dir = "artifacts/checkpoint-final"
    adapter_path = "artifacts/checkpoint-final"
    max_seq_len = 2048
    epochs = 5
    eval_steps = 20 if not DEBUG else 1
    warmup_ratio = 0.1
    learning_rate = 2e-4

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
    rng = np.random.default_rng(42)

    for idx, task in enumerate(dataset):
        file_name = filenames[idx]
        n_task = len(task)

        for _ in range(6):
            # 4개 그리드 랜덤 선택 (중복 허용)
            grids_idx = rng.choice(n_task, size=4, replace=True)
            train_grids = [task[i] for i in grids_idx[:3]]
            test_grids  = [task[i] for i in grids_idx[3:]]

            # test 포맷 정리
            test_inputs  = [{'input': g['input']} for g in test_grids]
            test_outputs = [g['output'] for g in test_grids]
            combined_tests = [
                {'input': g['input'], 'output': g['output']}
                for g in test_grids
            ]

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
    # from arc.arctest import ARCSolver

    solver = ARCSolver(token=token)

    run = wandb.init(
        project="arc",
        entity="dl-team14",
        name=cfg.model_name,
        config=cfg,
    )

    set_seed(1234567890)

    data_path = "/workspace/dataset"
    val_size = 0.1

    df = load_data(data_path)

    from datasets import Dataset
    dataset = Dataset.from_pandas(df).shuffle(42)

    train_val_split = dataset.train_test_split(test_size=val_size, seed=42)
    train_dataset = train_val_split['train']
    val_dataset = train_val_split['test']

    solver.train(train_dataset, val_dataset, cfg)
    
    run.finish()

if __name__ == "__main__":
    main()
