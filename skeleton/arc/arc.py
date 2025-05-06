from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from transformers import GenerationConfig
import torch
from typing import List
import numpy as np

from .utils import system_prompt, user_message_template1, user_message_template2, user_message_template3
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, TrainingArguments, pipeline
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import PeftModelForCausalLM

class cfg:
    adapter_path = "artifacts/checkpoint-final"
    output_dir = "artifacts/checkpoint-final"
    max_seq_len = 1024
    epochs = 5
    # max_steps = 1000
    eval_steps = 100
    warmup_ratio = 0.1
    learning_rate = 2e-4

class ARCSolver:
    """
    You should implement a `Solver` class for the project.
    """

    def __init__(self, token=None):
        """
        Args:
            token (str): a huggingface token for restricted models such as llama3
        """
        config_path = "artifacts/config/config.yml"
        model_id = "meta-llama/Llama-3.2-3B-Instruct"

        # Configure the BitsAndBytes settings for 4-bit quantization to reduce memory usage
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,  # Enable 4-bit quantization
            bnb_4bit_use_double_quant=True,  # Use double quantization for improved precision
            bnb_4bit_quant_type="nf4",  # Specify the quantization type
            bnb_4bit_compute_dtype=torch.float16,  # Set the computation data type
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True, # Allow the model to use custom code from the repository
            quantization_config=bnb_config, # Apply the 4-bit quantization configuration
            attn_implementation='sdpa', # Use scaled-dot product attention for better performance
            use_cache=False, # Disable caching to save memory
            device_map='auto', # Automatically map the model to available devices (e.g., GPUs)
            token=token,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.pixel_ids = [
            self.tokenizer.encode(str(i), add_special_tokens=False)[0] for i in range(10)
        ]
        self.sep = self.tokenizer.encode("\n", add_special_tokens=False)[0]
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

        sys = self.tokenizer.encode("<|begin_of_text|><|start_header_id|>system<|end_header_id|>" + "\n" + system_prompt, add_special_tokens=False)
        user = self.tokenizer.encode("<|start_header_id|>user<|end_header_id|>" + "\n" + user_message_template1 + "\n", add_special_tokens=False)
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
        user += self.tokenizer.encode("\n" + user_message_template3, add_special_tokens=False)


        messages = sys + user
        assis = self.tokenizer.encode("<|eot_id|><|start_header_id|>assistant<|end_header_id|>", add_special_tokens=False)

        if is_train:
            # attach labels to data
            output_test_data = datapoint['test'][0]['output']
            labels = self.format_grid(output_test_data)
            assis += labels
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

    def train(self, train_dataset, val_dataset=None):
        """
        Train a model with train_dataset.
        """
        self.model.gradient_checkpointing_enable()
        self.model.enable_input_require_grads()

        # 1. Format dataset
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

        # 2. Load LoRA Adapter
        print(f'\n*** Loading adapter from {cfg.adapter_path} ***')
        self.model = prepare_model_for_kbit_training(self.model)
        self.model = PeftModelForCausalLM.from_pretrained(
            self.model,
            cfg.adapter_path,
            device_map="auto",
            is_trainable=True,
        )

        # 3. Set training arguments
        print('\n*** Set training arguments ***')
        batch_size_kwargs = dict(
            per_device_train_batch_size=4,  # 4-16 should be fine for lora.
            gradient_accumulation_steps=4,
            per_device_eval_batch_size=4,
        )

        training_arguments = SFTConfig(
            output_dir=cfg.output_dir,          # output directory
            num_train_epochs=cfg.epochs,        # total number of training epochs
            # max_steps=cfg.max_steps,            # total number of training steps to perform
            warmup_ratio=cfg.warmup_ratio,      # number of warmup steps for learning rate scheduler
            learning_rate=cfg.learning_rate,    # learning rate
            lr_scheduler_type="linear",         # learning rate scheduler type
            optim="paged_adamw_8bit",           # optimizer to use

            do_eval=True,                       # whether to run evaluation on the validation set
            eval_strategy="steps",              # evaluation strategy to adopt during training
            save_steps=cfg.eval_steps,          # number of steps between two evaluations
            logging_steps=10,                   # number of steps between two logs
            eval_steps=cfg.eval_steps,          # number of steps between two evaluations
            log_level="debug",                  # set the logging level

            dataset_text_field="text",          # the name of the text field in the dataset
            max_seq_length=cfg.max_seq_len,     # maximum sequence length
            label_names=["labels"],

            **batch_size_kwargs
        )
        
        # 4. Set data collator
        print('\n*** Set data collator ***')
        data_collator = DataCollatorForCompletionOnlyLM(
            tokenizer=self.tokenizer,
            # instruction_template='<|start_header_id|>user<|end_header_id|>',
            response_template='<|start_header_id|>assistant<|end_header_id|>',
        )

        # 5. Train the model with SFTTrainer
        print('\n*** Train the model with SFTTrainer ***')
        trainer = SFTTrainer(
            model=self.model,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            args=training_arguments,
            # packing=True, # ValueError: You passed a `DataCollatorForCompletionOnlyLM` to the SFTTrainer. This is not compatible with the `packing` argument.
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
            grid = np.array(self.parse_grid(output))
            grid = grid[:x, :y]
            
        except Exception as e:
            grid = np.random.randint(0, 10, (x, y))

        return grid

    def prepare_evaluation(self):
        """
        Load pretrained weight, make model eval mode, etc.
        """
        self.model.load_adapter("artifacts/checkpoint-final")
        self.model.eval()


if __name__ == "__main__":
    solver = ARCSolver()




