import argparse
import os
import json
import pandas as pd
import numpy as np
import wandb
import yaml

import torch
from transformers import set_seed, BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, TrainingArguments, pipeline
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import LoraConfig, prepare_model_for_kbit_training, PeftModelForCausalLM
from unsloth import FastLanguageModel
from unsloth import UnslothTrainer as Trainer, unsloth_train, is_bfloat16_supported
from unsloth import UnslothTrainingArguments as TrainingArguments

from typing import List
from datasets import Dataset

from arc.utils import system_prompt, user_message_template1, user_message_template2, user_message_template3

class args_default:
    # default values for training
    # imported from baseline code

    model_id = "meta-llama/Llama-3.2-3B-Instruct"
    output_dir = "artifacts/checkpoint-final"
    adapter_path = "artifacts/checkpoint-final"
    config_path = "artifacts/config/config.yaml"

    max_seq_len = 2048

    dataset_len = 2000
    val_size = 0.01

    epochs = 1
    warmup_ratio = 0.1
    learning_rate = 2e-4
    lr_scheduler = "linear"
    optim = "paged_adamw_8bit"
    weight_decay = 0.01

    do_eval = True
    eval_strategy = "steps"
    eval_steps = 100
    save_steps = 100
    logging_steps = 10
    log_level = "debug"

    train_batch_size = 4
    grad_acc_steps = 8
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
    r = 64
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

    for task_idx in range(N):
        task = dataset[task_idx]
        file_name = filenames[task_idx]
        n_task = len(task)
        permutions = rng.permutation(n_task) # 랜덤으로 task 선택
        for j in range(0, n_task, 4):
            if j + 4 > n_task:
                break
            grids_idx =  permutions[j:j+4]
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
    parser.add_argument("--model-id", type=str, default=args_default.model_id, help="Model ID")
    parser.add_argument("--output-dir", type=str, default=args_default.output_dir, help="Output directory")
    parser.add_argument("--adapter-path", type=str, default=args_default.adapter_path, help="Adapter path")
    parser.add_argument("--config-path", type=str, default=args_default.config_path, help="Config path")
    
    parser.add_argument("--max-seq-len", type=int, default=args_default.max_seq_len, help="Max sequence length")
    
    # Model Configuration
    parser.add_argument("--attn-impl", type=str, default="flash", help="Attention implementation")
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

def format_grid(pixel_ids, sep, grid: List[List[int]]):
    """
    Format 2D grid into LLM input tokens

    Args:
        grid (List[List[int]]): 2D grid

    Returns:
        ids (List[int]): Token list for LLM
    """
    ids = []

    for row in grid:
        for col in row:
            ids.append(pixel_ids[col])
        ids.append(sep)
    return ids
    
def format_prompt(pixel_ids, sep, tokenizer, datapoint, is_train=False):
    """
    Args:
        datapoint (dict): contains training data, test input
        is_train (bool): whether the data is training/validation data or not
    
    Returns:
        prompt (dict): dictionary that contains input ids and additional informations
    """

    training_data = datapoint['train']
    input_test_data = datapoint['test'][0]['input']

    sys = tokenizer.encode("<|im_start|>system\n" + system_prompt + "<|im_end|>\n", add_special_tokens=False)
    user = tokenizer.encode("<|im_start|>user\n" + user_message_template1 + "\n", add_special_tokens=False)
    inp_desc = tokenizer.encode("input:\n", add_special_tokens=False)
    out_desc = tokenizer.encode("output:\n", add_special_tokens=False)
    for ex in training_data:
        inp = ex['input']
        out = ex['output']
        inp = format_grid(pixel_ids, sep, inp)
        out = format_grid(pixel_ids, sep, out)

        user += inp_desc
        user += inp
        user += out_desc
        user += out

    user += tokenizer.encode("\n" + user_message_template2 + "\n", add_special_tokens=False)

    user += inp_desc
    user += format_grid(pixel_ids, sep, input_test_data)
    user += tokenizer.encode("\n" + user_message_template3 + "<|im_end|>\n", add_special_tokens=False)


    messages = sys + user
    assis = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)

    if is_train:
        # attach labels to data
        output_test_data = datapoint['test'][0]['output']
        labels = format_grid(pixel_ids, sep, output_test_data)
        assis += labels
        assis += tokenizer.encode("<|im_end|>\n", add_special_tokens=False)
    messages += assis
    
    attention_mask = [1] * len(messages)

    if is_train:
        return {
            "input_ids": messages,
            "attention_mask": attention_mask,
        }
    else:
        return {
            "input_ids": messages,
            "attention_mask": attention_mask,

            # Required for post-processing the shape of LLM-generated grid
            # Hence these fields are not used in training
            "input": input_test_data,
            "train": training_data,
        }

def main():
    '''
    Main function to run the training process.
    '''
    token = os.environ.get("HF_TOKEN", None)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    args = parse_args()
    export_args(args)

    set_seed(1234567890)

    data_path = "/workspace/dataset"
    df = load_data(data_path, args)
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

    print("*** Setup model and tokenizer with config ***")
    
    # Configure the BitsAndBytes settings for 4-bit quantization to reduce memory usage
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,  # Enable 4-bit quantization
        bnb_4bit_use_double_quant=True,  # Use double quantization for improved precision
        bnb_4bit_quant_type="nf4",  # Specify the quantization type
        bnb_4bit_compute_dtype=torch.float16,  # Set the computation data type
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        trust_remote_code=True, # Allow the model to use custom code from the repository
        quantization_config=bnb_config, # Apply the 4-bit quantization configuration
        # attn_implementation=args.attn_impl, # Use scaled-dot product attention for better performance
        use_cache=args.use_cache, # Disable caching to save memory
        device_map='auto', # Automatically map the model to available devices (e.g., GPUs)
        token=token,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_id, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    pixel_ids = [
        tokenizer.encode(str(i), add_special_tokens=False)[0] for i in range(10)
    ]
    sep = tokenizer.encode("\n", add_special_tokens=False)[0]

    # Setup model and tokenizer with config
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    # Load LoRA Adapter
    peft_config = None
    if args.load_adapter:
        print(f'\n*** Loading adapter from {args.adapter_path} ***')
        model = prepare_model_for_kbit_training(model)
        model = PeftModelForCausalLM.from_pretrained(
            model,
            args.adapter_path,
            device_map="auto",
            is_trainable=True,
        )
    else:
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=args.lora_target_modules,
            lora_dropout=args.lora_dropout,
            bias=args.lora_bias,
            task_type=args.lora_task_type,
        )
        model = prepare_model_for_kbit_training(model)

    # Format dataset
    print('*** Format dataset ***')
    train_dataset = train_dataset.map(
        lambda x: format_prompt(pixel_ids, sep, tokenizer, x, is_train=True),
        remove_columns=train_dataset.column_names,
    )

    val_dataset = val_dataset.map(
        lambda x: format_prompt(pixel_ids, sep, tokenizer, x, is_train=True),
        remove_columns=val_dataset.column_names,
    )

    # Set data collator
    print('\n*** Set data collator ***')
    data_collator = DataCollatorForCompletionOnlyLM(
        tokenizer=tokenizer,
        # instruction_template='<|im_start|>user\n',
        response_template='<|im_start|>assistant\n',
    )

    # Set training arguments
    print('\n*** Set training arguments ***')
    batch_size_kwargs = dict(
        per_device_train_batch_size=args.train_batch_size,  # 4-16 should be fine for lora.
        gradient_accumulation_steps=args.grad_acc_steps,
        per_device_eval_batch_size=args.eval_batch_size,
    )

    training_arguments = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        # max_steps=args.max_steps,
        warmup_ratio=args.warmup_ratio,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler,
        optim=args.optim,
        weight_decay=args.weight_decay, 

        do_eval=args.do_eval,
        eval_strategy=args.eval_strategy,
        eval_steps=args.eval_steps,
        save_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        log_level=args.log_level,

        max_seq_length=args.max_seq_len,
        label_names=["labels"],
        report_to="wandb" if args.wandb else "none",

        **batch_size_kwargs
    )

    # Train the model with SFTTrainer
    print('\n*** Train the model with SFTTrainer ***')
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        peft_config=peft_config,
        data_collator=data_collator,
        args=training_arguments,
    )

    trainer.train()

    if args.wandb:
        # Finish the run
        wandb.finish()

if __name__ == "__main__":
    main()