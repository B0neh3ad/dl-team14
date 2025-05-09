import argparse
import os

from transformers import set_seed
import json
import pandas as pd
import numpy as np
import wandb
import yaml
import argparse

class args_default:
    # default values for training
    # imported from baseline code

    model_id = "meta-llama/Llama-3.2-3B-Instruct"
    output_dir = "artifacts/checkpoint-final"
    adapter_path = "artifacts/checkpoint-final"
    config_path = "artifacts/config/config.yaml"

    max_seq_len = 2048

    dataset_len = 2000
    val_size = 0.1

    epochs = 5
    warmup_ratio = 0.1
    learning_rate = 2e-4
    lr_scheduler = "linear"
    optim = "paged_adamw_8bit"
    weight_decay = 0.01

    do_eval = True
    eval_strategy = "steps"
    eval_steps = 20
    save_steps = 20
    logging_steps = 10
    log_level = "debug"

    train_batch_size = 4
    grad_acc_steps = 4
    eval_batch_size = 4

    wandb = True

    debug = False

class adapter_config_default:
    # default values for LoRA
    # imported from checkpoint-final/adapter_config.json (baseline)

    alpha_pattern = {}
    auto_mapping = None
    base_model_name_or_path = args_default.model_id
    bias = "none"
    corda_config = None
    eva_config = None
    exclude_modules = None
    fan_in_fan_out = False
    inference_mode = True
    init_lora_weights = True
    layer_replication = None
    layers_pattern = None
    layers_to_transform = None
    loftq_config = {}
    lora_alpha = 64
    lora_bias = False
    lora_dropout = 0.05
    megatron_config = None
    megatron_core = "megatron.core"
    modules_to_save = None
    peft_type = "LORA"
    r = 4
    rank_pattern = {}
    revision = None
    target_modules = [
        "up_proj",
        "down_proj",
        "gate_proj",
        "o_proj",
        "v_proj",
        "k_proj",
        "q_proj"
    ]
    task_type = "CAUSAL_LM"
    trainable_token_indices = None
    use_dora = False
    use_rslora = False
    
def load_data(base_dir, args=None):
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
    MAX_LEN = args.dataset_len if args else 2000
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

def parse_args():
    '''
    Parse command line arguments.
    '''    
    print("Parsing arguments...")

    parser = argparse.ArgumentParser(description="Train the model.")
    parser.add_argument("--model_id", type=str, default=args_default.model_id, help="Model ID")
    parser.add_argument("--output_dir", type=str, default=args_default.output_dir, help="Output directory")
    parser.add_argument("--adapter_path", type=str, default=args_default.adapter_path, help="Adapter path")
    parser.add_argument("--config_path", type=str, default=args_default.config_path, help="Config path")
    
    parser.add_argument("--max_seq_len", type=int, default=args_default.max_seq_len, help="Max sequence length")
    
    # Model Configuration
    parser.add_argument("--attn-impl", type=str, default="sdpa", help="Attention implementation")
    parser.add_argument("--use-cache", action="store_true", help="Use cache")

    # Dataset Configuration
    parser.add_argument("--dataset_len", type=int, default=args_default.dataset_len, help="Dataset length")
    parser.add_argument("--val_size", type=float, default=args_default.val_size, help="Validation size")

    # Training Configuration
    parser.add_argument("--epochs", type=int, default=args_default.epochs, help="Number of epochs")
    parser.add_argument("--warmup_ratio", type=float, default=args_default.warmup_ratio, help="Warmup ratio")
    parser.add_argument("--learning_rate", type=float, default=args_default.learning_rate, help="Learning rate")
    parser.add_argument("--lr_scheduler", type=str, default=args_default.lr_scheduler, help="Learning rate scheduler")
    parser.add_argument("--optim", type=str, default=args_default.optim, help="Optimizer")
    parser.add_argument("--weight_decay", type=float, default=args_default.weight_decay, help="Weight decay")

    parser.add_argument("--do_eval", action="store_true", help="Evaluate the model")
    parser.add_argument("--eval_strategy", type=str, default=args_default.eval_strategy, help="Evaluation strategy")
    parser.add_argument("--eval_steps", type=int, default=args_default.eval_steps, help="Evaluation steps")
    parser.add_argument("--save_steps", type=int, default=args_default.save_steps, help="Save steps")
    parser.add_argument("--logging_steps", type=int, default=args_default.logging_steps, help="Logging steps")
    parser.add_argument("--log-level", type=str, default=args_default.log_level, help="Log level")

    parser.add_argument("--train_batch_size", type=int, default=args_default.train_batch_size, help="Batch size per device for training")
    parser.add_argument("--grad_acc_steps", type=int, default=args_default.grad_acc_steps, help="Number of gradient accumulation steps")
    parser.add_argument("--eval_batch_size", type=int, default=args_default.eval_batch_size, help="Batch size per device for evaluation")

    # LoRA Configuration
    parser.add_argument("--lora-r", type=int, default=adapter_config_default.r, help="Rank for LoRA")
    parser.add_argument("--lora_alpha", type=int, default=adapter_config_default.lora_alpha, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=adapter_config_default.lora_dropout, help="LoRA dropout")
    parser.add_argument("--use-rslora", action="store_true", help="Use RSLORA")
    parser.add_argument("--use-dora", action="store_true", help="Use DORA")

    # WandB Configuration
    parser.add_argument("--wandb", action="store_true", default=args_default.wandb, help="Use WandB for logging")

    # Debug mode
    parser.add_argument("--debug", action="store_true", default=args_default.debug, help="Debug mode")

    args = parser.parse_args()

    if args.debug:
        args.eval_steps = 1
        args.epochs = 1
        args.learning_rate = 1e-4
        args.max_seq_len = 2048
        args.output_dir = "artifacts/checkpoint-debug"

        args.dataset_len = 100

        args.train_batch_size = 1
        args.grad_acc_steps = 1
        args.eval_batch_size = 1
    
    return args

def export_args(args):
    # Convert args to dictionary
    args_dict = vars(args)

    # Export to YAML file
    with open(args.config_path, 'w') as f:
        yaml.dump(args_dict, f)

    print(f"Arguments saved to {args.config_path}")

def main():
    '''
    Main function to run the training process.
    '''
    token = os.environ.get("HF_TOKEN", None)

    args = parse_args()
    export_args(args)

    from arc import ARCSolver
    solver = ARCSolver(token=token)

    set_seed(1234567890)

    data_path = "/workspace/dataset"
    df = load_data(data_path, args)

    from datasets import Dataset
    dataset = Dataset.from_pandas(df).shuffle(42)

    train_val_split = dataset.train_test_split(test_size=args.val_size, seed=42)
    train_dataset = train_val_split['train']
    val_dataset = train_val_split['test']

    run = None
    if args.wandb:
        run = wandb.init(
            project="arc",
            entity="dl-team14",
            name=args.model_id,
            config=args,
        )

    solver.train(train_dataset, val_dataset, args)

    if args.wandb:
        # Finish the run
        wandb.finish()

        # TODO: Save the model to wandb

if __name__ == "__main__":
    main()
