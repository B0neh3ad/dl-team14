from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from transformers import GenerationConfig
import torch
from typing import List
import numpy as np

from .utils import system_prompt, user_message_template1, user_message_template2, user_message_template3
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, TrainingArguments, pipeline
from unsloth import FastLanguageModel
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from unsloth import UnslothTrainer as Trainer, unsloth_train, is_bfloat16_supported
from unsloth import UnslothTrainingArguments as TrainingArguments

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

        # Configure the BitsAndBytes settings for 4-bit quantization to reduce memory usag
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(model_id, max_seq_length=cfg.max_seq_len, load_in_4bit=True)

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

    def train(self, train_dataset, val_dataset=None, cfg=None):
        """
        Train a model with train_dataset.
        """
        FastLanguageModel.for_training(self.model)

        # Load model with LoRA adapter
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            target_modules=[
                'q_proj','k_proj','v_proj','o_proj',
                'gate_proj','up_proj','down_proj',
            ],
            r=32,
            lora_alpha=64,
            lora_dropout=0.0,
            bias="none",
            use_gradient_checkpointing=True,
            random_state=42,
            use_rslora=True,
        )

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
        training_arguments = TrainingArguments(
            output_dir=cfg.output_dir,
            num_train_epochs=cfg.epochs,
            warmup_ratio=0.25, # cfg.warmup_ratio
            learning_rate=1e-4, # cfg.learning_rate
            lr_scheduler_type='cosine',
            optim="adamw_8bit",

            do_eval=True,
            eval_strategy='steps',
            save_steps=cfg.eval_steps,
            logging_steps=10,
            eval_steps=cfg.eval_steps,
            log_level="debug",

            dataset_text_field="text",
            max_seq_length=cfg.max_seq_len,
            label_names=["labels"],

            per_device_train_batch_size=4,
            gradient_accumulation_steps=2,
            per_device_eval_batch_size=4,

            embedding_learning_rate=1e-5,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            weight_decay=0.00,
            seed=42,
        )

        # Train the model with Trainer
        trainer = Trainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            args=training_arguments,
            packing=False,
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
        self.model.load_adapter(cfg.output_dir)
        self.model.eval()


if __name__ == "__main__":
    for name, module in self.model.named_modules():
        print(name)

   




