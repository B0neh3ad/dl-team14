import argparse
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from transformers import GenerationConfig
import torch
from typing import List
import numpy as np
import yaml

from .utils import system_prompt, user_message_template1, user_message_template2, user_message_template3
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, TrainingArguments, pipeline
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import PeftModelForCausalLM

class ARCSolver:
    """
    You should implement a `Solver` class for the project.
    """

    def __init__(self, token=None):
        """
        Args:
            token (str): a huggingface token for restricted models such as llama3
        """
        self.token = token
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def parse_grid(self, ids: List[int]):
        """
        Parse LLM generated sequence into ARC grid format

        Args:
            ids (List[int]): LLM generated token list

        Returns:
            grid (List[List[int]]): parsed 2D grid
        """
        grid = []
        row = []
        inv_map = {k: i for i, k in enumerate(self.pixel_ids)}
        
        for idx in ids:
            if idx == self.sep:
                if len(row) > 0:
                    grid.append(row.copy())
                    row.clear()
            else:
                row.append(inv_map.get(idx, 0))
        return grid

    def format_grid(self, grid: List[List[int]]):
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
                ids.append(self.pixel_ids[col])
            ids.append(self.sep)
        return ids

    def format_prompt(self, datapoint, is_train=False):
        """
        Args:
            datapoint (dict): contains training data, test input
            is_train (bool): whether the data is training/validation data or not
        
        Returns:
            prompt (dict): dictionary that contains input ids and additional informations
        """

        training_data = datapoint['train']
        input_test_data = datapoint['test'][0]['input']

        sys = self.tokenizer.encode("<|im_start|>system\n" + system_prompt + "<|im_end|>\n", add_special_tokens=False)
        user = self.tokenizer.encode("<|im_start|>user\n" + user_message_template1 + "\n", add_special_tokens=False)
        inp_desc = self.tokenizer.encode("input:\n", add_special_tokens=False)
        out_desc = self.tokenizer.encode("output:\n", add_special_tokens=False)
        for ex in training_data:
            inp = ex['input']
            out = ex['output']
            inp = self.format_grid(inp)
            out = self.format_grid(out)

            user += inp_desc
            user += inp
            user += out_desc
            user += out

        user += self.tokenizer.encode("\n" + user_message_template2 + "\n", add_special_tokens=False)

        user += inp_desc
        user += self.format_grid(input_test_data)
        user += self.tokenizer.encode("\n" + user_message_template3 + "<|im_end|>\n", add_special_tokens=False)


        messages = sys + user
        assis = self.tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)

        if is_train:
            # attach labels to data
            output_test_data = datapoint['test'][0]['output']
            labels = self.format_grid(output_test_data)
            assis += labels
            assis += self.tokenizer.encode("<|im_end|>\n", add_special_tokens=False)
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

    def setup(self, args):
        print("*** Setup model and tokenizer with config ***")
        
        # Configure the BitsAndBytes settings for 4-bit quantization to reduce memory usage
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,  # Enable 4-bit quantization
            bnb_4bit_use_double_quant=True,  # Use double quantization for improved precision
            bnb_4bit_quant_type="nf4",  # Specify the quantization type
            bnb_4bit_compute_dtype=torch.float16,  # Set the computation data type
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            trust_remote_code=True, # Allow the model to use custom code from the repository
            quantization_config=bnb_config, # Apply the 4-bit quantization configuration
            attn_implementation=args.attn_impl, # Use scaled-dot product attention for better performance
            use_cache=args.use_cache, # Disable caching to save memory
            device_map='auto', # Automatically map the model to available devices (e.g., GPUs)
            token=self.token,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(args.model_id, token=self.token)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.pixel_ids = [
            self.tokenizer.encode(str(i), add_special_tokens=False)[0] for i in range(10)
        ]
        self.sep = self.tokenizer.encode("\n", add_special_tokens=False)[0]


    def train(self, train_dataset, val_dataset=None, args=None):
        """
        Train a model with train_dataset.
        Args:
            train_dataset (Dataset): training dataset
            val_dataset (Dataset): validation dataset
            args (Class): configuration for training
        
        Below code is imported from
        https://github.com/ironbar/arc24/blob/main/notebooks/003_llm_fine-tuning_on_arc_tasks.ipynb
        """

        # Setup model and tokenizer with config
        self.setup(args)
        self.model.gradient_checkpointing_enable()
        self.model.enable_input_require_grads()

        # Load LoRA Adapter
        peft_config = None
        if args.load_adapter:
            print(f'\n*** Loading adapter from {args.adapter_path} ***')
            self.model = prepare_model_for_kbit_training(self.model)
            self.model = PeftModelForCausalLM.from_pretrained(
                self.model,
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
            self.model = prepare_model_for_kbit_training(self.model)

        # Format dataset
        print('*** Format dataset ***')
        # TODO: implement batched processing
        train_dataset = train_dataset.map(
            lambda x: self.format_prompt(x, is_train=True),
            remove_columns=train_dataset.column_names,
        )

        val_dataset = val_dataset.map(
            lambda x: self.format_prompt(x, is_train=True),
            remove_columns=val_dataset.column_names,
        )

        # Set data collator
        print('\n*** Set data collator ***')
        data_collator = DataCollatorForCompletionOnlyLM(
            tokenizer=self.tokenizer,
            # instruction_template='<|start_header_id|>user<|end_header_id|>',
            response_template='<|start_header_id|>assistant<|end_header_id|>',
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
            model=self.model,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            peft_config=peft_config,
            data_collator=data_collator,
            args=training_arguments,
        )

        trainer.train()

    def predict(self, examples, questions_input):
        """
        A single example of test data is given.
        You should predict 2D grid (List[List[int]] or np.ndarray)

        Args:
            examples (List[dict]): List of training examples,
                each list element is a dictionary that contains "input" and "output"
                for example,
                [
                    {
                        "input": [[1,2],[3,4]],
                        "output": [[4,5],[6,7]],
                    },
                    {
                        "input": [[0,1],[2,3]],
                        "output": [[3,4],[5,6]],
                    }
                ]
            questions_input (List[List[int]]): A 2d grid,
                which is a input for a given question
        Returns:
            output (List[List[int]]): A 2d grid,
                which is the output of given input question.
        """
        datapoint = {
            "train": examples,
            "test": [
                {
                    "input": questions_input
                }
            ]
        }

        prompt = self.format_prompt(datapoint)
        input_ids = torch.tensor(prompt['input_ids'], dtype=torch.long).to(self.device).view(1, -1)

        config = GenerationConfig(
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=150,
        )

        output = self.model.generate(
            input_ids=input_ids,
            generation_config=config,
        ).squeeze().cpu()
        N_prompt = input_ids.numel()

        output = output[N_prompt:].tolist()
        train_input = np.array(prompt['train'][0]['input'])
        train_output = np.array(prompt['train'][0]['output'])
        test_input = np.array(prompt['input'])

        # LLM-generated grid may have wrong shape
        # So adjust shape by input-output pairs
        if train_input.shape == train_output.shape:
            x, y = test_input.shape
        else:
            x = (train_output.shape[0] // train_input.shape[0]) * test_input.shape[0]
            y = (train_output.shape[1] // train_input.shape[1]) * test_input.shape[1]

        try:
            print(f"output: {output}")
            grid = np.array(self.parse_grid(output))
            # grid = grid[:x, :y]
            
        except Exception as e:
            grid = np.random.randint(0, 10, (x, y))

        return grid

    def prepare_evaluation(self):
        """
        Load pretrained weight, make model eval mode, etc.
        """
        # Load config yaml file
        # NOTE: You should locate config file in this path!
        config_path = "artifacts/config/config-qwen.yaml"
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)
        
        args = argparse.Namespace(**config_dict)

        # Setup model and tokenizer with config
        self.setup(args)

        self.model.load_adapter(args.output_dir)
        self.model.eval()


if __name__ == "__main__":
    solver = ARCSolver()




