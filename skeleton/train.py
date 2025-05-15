import argparse
import os

from transformers import set_seed
import json
import pandas as pd
import numpy as np
import wandb
import yaml
import argparse
from arc.dataloader import ArcDataLoader
from datasets import Dataset

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
    


def parse_args():
    '''
    Parse command line arguments.
    '''    
    print("Parsing arguments...")

    parser = argparse.ArgumentParser(description="Train the model.")
    parser.add_argument("--model-id", type=str, default=args_default.model_id, help="Model ID")
    parser.add_argument("--output-dir", type=str, default=args_default.output_dir, help="Output directory")
    parser.add_argument("--adapter-path", type=str, default=args_default.adapter_path, help="Adapter path")
    parser.add_argument("--config-path", type=str, default=args_default.config_path, help="Config path")
    
    parser.add_argument("--max-seq-len", type=int, default=args_default.max_seq_len, help="Max sequence length")
    
    # Model Configuration
    parser.add_argument("--attn-impl", type=str, default="sdpa", help="Attention implementation")
    parser.add_argument("--use-cache", action="store_true", help="Use cache")

    # Dataset Configuration
    parser.add_argument("--dataset-len", type=int, default=args_default.dataset_len, help="Dataset length")
    parser.add_argument("--val-size", type=float, default=args_default.val_size, help="Validation size")

    # Training Configuration
    parser.add_argument("--epochs", type=int, default=args_default.epochs, help="Number of epochs")
    parser.add_argument("--warmup-ratio", type=float, default=args_default.warmup_ratio, help="Warmup ratio")
    parser.add_argument("--learning-rate", type=float, default=args_default.learning_rate, help="Learning rate")
    parser.add_argument("--lr-scheduler", type=str, default=args_default.lr_scheduler, help="Learning rate scheduler")
    parser.add_argument("--optim", type=str, default=args_default.optim, help="Optimizer")
    parser.add_argument("--weight-decay", type=float, default=args_default.weight_decay, help="Weight decay")

    parser.add_argument("--do-eval", action="store_true", help="Evaluate the model")
    parser.add_argument("--eval-strategy", type=str, default=args_default.eval_strategy, help="Evaluation strategy")
    parser.add_argument("--eval-steps", type=int, default=args_default.eval_steps, help="Evaluation steps")
    parser.add_argument("--save-steps", type=int, default=args_default.save_steps, help="Save steps")
    parser.add_argument("--logging-steps", type=int, default=args_default.logging_steps, help="Logging steps")
    parser.add_argument("--log-level", type=str, default=args_default.log_level, help="Log level")

    parser.add_argument("--train-batch-size", type=int, default=args_default.train_batch_size, help="Batch size per device for training")
    parser.add_argument("--grad-acc-steps", type=int, default=args_default.grad_acc_steps, help="Number of gradient accumulation steps")
    parser.add_argument("--eval-batch-size", type=int, default=args_default.eval_batch_size, help="Batch size per device for evaluation")

    # LoRA Configuration
    parser.add_argument("--load-adapter", action="store_true", default=False, help="Load adapter from path")
    parser.add_argument("--lora-r", type=int, default=adapter_config_default.r, help="Rank for LoRA")
    parser.add_argument("--lora-alpha", type=int, default=adapter_config_default.lora_alpha, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=adapter_config_default.lora_dropout, help="LoRA dropout")
    parser.add_argument("--lora-bias", type=str, default=adapter_config_default.bias, help="LoRA bias")
    parser.add_argument("--lora-target-modules", nargs='+', type=str, default=adapter_config_default.target_modules, help="LoRA target modules")
    parser.add_argument("--lora-task-type", type=str, default=adapter_config_default.task_type, help="LoRA task type")
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

    dataloader = ArcDataLoader.load_from_json(data_path)
    dataset_list = dataloader.make_dataset(4, solver.fmt_opts, True)
    dataset = Dataset.from_list(dataset_list)


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

    print(train_dataset)

#    solver.train(train_dataset, val_dataset, args)

    if args.wandb:
        # Finish the run
        wandb.finish()

        # TODO: Save the model to wandb

if __name__ == "__main__":
    main()
