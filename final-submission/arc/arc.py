import argparse
import os
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from transformers import GenerationConfig
import torch
from typing import List
import numpy as np
import yaml
import re

from unsloth import FastLanguageModel
from .arc_loader import ArcDataset
from .model_tools import load_unsloth_4bit
from .inference_tools import inference_run
from .selection import EvalTool

from datasets import Dataset
from .model_tools import InputMaskingDataCollator
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
        self.token = token
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.fmt_opts = dict(
            preprompt='ABCDEFGHJKLMNPQRSTUVWXYZabcdefghjklmnpqrstuvwxyz',
            query_beg='I',
            reply_beg='\n+/-=O',
            reply_end='\n',
            lines_sep='\n',
            max_tokens=8192,
        )

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
    
    def _init_ttt_adapter(self):
    
        if hasattr(self, "_ttt_ready"):           # 이미 한 번 만들었으면 패스
            return

        self.model = FastLanguageModel.get_peft_model(
            model=self.model,
            target_modules=['q_proj','k_proj','v_proj','o_proj',
                        'gate_proj','up_proj','down_proj'],
            r=128,              # 필요하면 16·64 등 조정
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            random_state=42,
            use_rslora=True
        )
        self._ttt_ready = True

    def _zero_ttt_weights(self):
        for p in self.model.parameters():
            if getattr(p, "is_lora", False):
                p.data.zero_()

    def setup(self, base_model):
        print("*** Setup model and tokenizer with config ***")
        self.model, self.tokenizer = load_unsloth_4bit(base_model)
        self.fmt_opts = dict(
            preprompt='ABCDEFGHJKLMNPQRSTUVWXYZabcdefghjklmnpqrstuvwxyz',
            query_beg='I',
            reply_beg='\n+/-=O',
            reply_end='\n' + self.tokenizer.eos_token,
            lines_sep='\n',
            max_tokens=128000,
        )

    def find_algorithm(self, train_inputs, train_outputs):
        """
        Find the algorithm that solves the given training data.

        Args:
            train_inputs (List[List[List[int]]]): Three 2D grids,
                which is a inputs for a given question
            train_outputs (List[List[List[int]]]): Three 2D grids,
                which is the outputs of given input question.

        Returns:
            algorithm (function): A function that takes a 2D grid as input and returns a 2D grid as output.
        """
        import importlib, inspect
        tasksolver = importlib.import_module('.dsl_solver', package='.arc')

        train_inputs = [tuple(tuple(line) for line in train_inputs[i]) for i in range(len(train_inputs))]
        train_outputs = [tuple(tuple(line) for line in train_outputs[i]) for i in range(len(train_outputs))]
        all_functions = []
        error_cnt = 0
        for name, func in inspect.getmembers(tasksolver, inspect.isfunction):
            if name.startswith('solve_'):
                all_functions.append(func)

        # apply each functions to train_input and check result
        for func in all_functions:
            try:
                results = [func(train_input) for train_input in train_inputs]

                # see if result is equal to train_output
                if results == train_outputs:
                    # print(f"====Found function {func.__name__}====\nresult:\n{results}\n train_outputs:\n{train_outputs}")
                    return func
            except Exception as e:
                # skip if error
                error_cnt += 1
                pass

        # if no function found, return None
        # print(f"====No function found====\nfunctions_cnt: {len(all_functions)}, error_cnt: {error_cnt}, wrong_cnt: {len(all_functions) - error_cnt}")
        return None
    
    def test_time_train(self, ttt_ds):
        """
        Test-time train the model with the given training data.

        Args:
            ttt_ds (ArcDataset): The training dataset to use for test-time training.
        """

        ttt_ds_aug = ttt_ds.remove_test_data().repeat(n=48, seed=42).augment(**self.ttt_aug_opts)
        ttt_ds_as_list = ttt_ds_aug.as_list(len_name='text', **self.fmt_opts)

        try:
            # _flag_for_generation 속성이 없어도 계속 진행
            FastLanguageModel.for_training(self.model)
            trainer = Trainer(
                model=self.model,
                tokenizer=self.tokenizer,
                train_dataset=Dataset.from_list(ttt_ds_as_list),
                dataset_text_field="text",
                max_seq_length=self.fmt_opts['max_tokens'],
                data_collator=InputMaskingDataCollator(
                    instruction_template=self.fmt_opts['query_beg'],
                    response_template=self.fmt_opts['reply_beg'],
                    mlm=False,
                    tokenizer=self.tokenizer,
                    mask_first_n_examples=0,
                ),
                args=TrainingArguments(
                    per_device_train_batch_size=4,
                    gradient_accumulation_steps=2,
                    warmup_ratio=0.0,
                    num_train_epochs=1,
                    learning_rate=5e-5,
                    embedding_learning_rate=1e-5,
                    fp16=not is_bfloat16_supported(),
                    bf16=is_bfloat16_supported(),
                    logging_steps=1,
                    optim="adamw_8bit",
                    weight_decay=0.00,
                    lr_scheduler_type='cosine',
                    seed=42,
                    output_dir='tmp_output',
                    save_strategy='no',
                    report_to='none',
                ),
            )

            trainer_stats = unsloth_train(trainer)
        except AttributeError as e:
            if '_flag_for_generation' in str(e):
                print("Note: Model was already in training mode or flag not found. Continuing...")
            else:
                raise  # 다른 AttributeError는 다시 발생시킴
    
    def infer(self, base, ds, min_prob=0.9):
        """
        Run inference on the given dataset.

        Args:
            ds (ArcDataset): The dataset to run inference on.

        Returns:
            inference_results (dict): The results of the inference.
        """
        FastLanguageModel.for_inference(self.model)
        inference_results = inference_run(
            model_tok=(self.model, self.tokenizer),
            fmt_opts=self.fmt_opts,
            dataset=ds,
            min_prob=min_prob,
            aug_score_opts=self.infer_aug_opts,
            callback=self.eval_tool.process_result,
        )

        x, y = np.random.randint(1, 10), np.random.randint(1, 10)
        best_output = np.random.randint(0, 10, (x, y))
        best_score  = float('-inf')

        for aug_idx, guesses in enumerate(inference_results[base]):
            # guesses 비어있으면 continue
            if len(guesses) == 0:
                continue
            # n_guesses=1 이면 guesses 리스트에 단 하나만 들어있어
            guess = guesses[0]
            score = guess['scores_alg'][self.eval_tool.sorting_algo]  # 정렬에 쓰는 스코어 인덱스
            if score > best_score:
                best_score  = score
                best_output = guess['output']
                
        return best_score, best_output


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
        base = 'mem'
        challenge = {
            base: {
                'train': examples,                # List[dict]
                'test' : [{'input': questions_input}]
            }
        }
        keys = [f'{base}_0']
        
        ttt_ds = ds = ArcDataset(challenge=challenge, keys=keys, is_orig=True)
        for i in range(1):
            print(f"{i+1}th infer")
            self.infer_aug_opts["seed"] = self.our_lucky_seed[i];
            best_score, best_output = self.infer(base, ds.augment(**self.infer_aug_opts), min_prob=(0.9 - (0.1) * i))
            if(best_score != float('-inf')):
                return best_output;
    
        
        self.model.enable_adapter_layers() 
        self.test_time_train(ttt_ds)
        for i in range(5):
            print(f"{i+1}th infer")
            self.infer_aug_opts["seed"] = self.our_lucky_seed[i];
            best_score, best_output = self.infer(base, ds.augment(**self.infer_aug_opts), min_prob=(0.9 - (0.2) * i))
            if(best_score != float('-inf')):
                return best_output;
        self._zero_ttt_weights()
        self.model.disable_adapter_layers()


        return best_output;


    def prepare_evaluation(self):
        """
        Load pretrained weight, make model eval mode, etc.
        """
        base_model = 'Qwen2.5-3B-Instruct-merged'
        save_model_path = os.path.join('artifacts', base_model)

        # Setup model and tokenizer with config
        self.setup(save_model_path)

        self._init_ttt_adapter()
        self._zero_ttt_weights()
        self.infer_aug_opts = dict(tp='all', rt='all', perm=True, shfl_ex=True, seed=10000)
        self.eval_tool = EvalTool(n_guesses=1)
        self.ttt_aug_opts = dict(tp=True, rt=True, perm=True, shfl_ex=True, seed=0)
        self.our_lucky_seed = [42, 627, 801, 820, 526]


if __name__ == "__main__":
    solver = ARCSolver()
