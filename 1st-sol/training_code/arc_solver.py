import argparse
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from transformers import GenerationConfig
import torch
from typing import List
import numpy as np
import yaml
import re

from unsloth import FastLanguageModel
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

    def setup(self, args):
        print("*** Setup model and tokenizer with config ***")
        self.model, self.tokenizer = load_unsloth_4bit(args.model_id, token=self.token)
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
        test_input = np.array(prompt['input'])

        # 1. 토큰 리스트 디코딩
        decoded_text = self.tokenizer.decode(output[:10], skip_special_tokens=True)

        # 2. 숫자 추출
        pattern = r'\((\d+\.?\d*),\s*(\d+\.?\d*)\)'
        match = re.search(pattern, decoded_text)

        if match:
            width_str, height_str = match.groups()
            width = int(width_str)
            height = int(height_str)

        else:
            width, height = test_input.shape  # 패턴 미발견 시 처리
        

        try:
            grid = np.array(self.parse_grid(output))
            grid = grid[:width, :height]
            
        except Exception as e:
            grid = np.random.randint(0, 10, (width, height))

        return grid

    def prepare_evaluation(self):
        """
        Load pretrained weight, make model eval mode, etc.
        """
        # Load config yaml file
        # NOTE: You should locate config file in this path!
        config_path = "artifacts/config/config.yaml"
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)
        
        args = argparse.Namespace(**config_dict)

        # Setup model and tokenizer with config
        self.setup(args)

        FastLanguageModel.for_inference(self.model)


if __name__ == "__main__":
    solver = ARCSolver()




