import argparse
import os
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from transformers import GenerationConfig
import torch
from typing import List
import numpy as np
import yaml
import re
from diskcache import Cache

from unsloth import FastLanguageModel
from arc_loader import ArcDataset
from model_tools import load_unsloth_4bit
from inference_tools import inference_run
from selection import EvalTool

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

    def setup(self, base_model):
        print("*** Setup model and tokenizer with config ***")
        self.model, self.tokenizer = load_unsloth_4bit(base_model, token=self.token)
        self.fmt_opts = dict(
            preprompt='ABCDEFGHJKLMNPQRSTUVWXYZabcdefghjklmnpqrstuvwxyz',
            query_beg='I',
            reply_beg='\n+/-=O',
            reply_end='\n' + self.tokenizer.eos_token,
            lines_sep='\n',
            max_tokens=128000,
        )

    def train(self, train_dataset, val_dataset=None, args=None):
        """
        Refer to `run_finetuning_...-arc.py` for training code.
        """
        pass

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

        # TODO: Implement the predict function

        # 1. foramt datapoint and augment
        # 2. perform inference for augmented datapoint
        # 3. 
        grid = []

        return grid

    def prepare_evaluation(self):
        """
        Load pretrained weight, make model eval mode, etc.
        """
        base_model = 'Qwen2.5-3B-Instruct-merged'

        output_path = 'output_evaluation_Llama-arc_without_ttt'
        save_model_path = os.path.join('finetuned_models', base_model)
        inference_cache = os.path.join(output_path, 'inference_cache')

        # Setup model and tokenizer with config
        self.setup(base_model)

        FastLanguageModel.for_inference(self.model)
        self.infer_aug_opts = dict(tp='all', rt='all', perm=True, shfl_ex=True, seed=10000)
        self.model_cache = Cache(inference_cache).memoize(typed=True, ignore=set(['model_tok', 'guess']))
        self.eval_tool = EvalTool(n_guesses=1)


if __name__ == "__main__":
    solver = ARCSolver()




