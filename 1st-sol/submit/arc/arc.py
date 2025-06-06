from unsloth import FastLanguageModel

import os
import torch
from typing import List
import numpy as np

from .arc_loader import ArcDataset
from .model_tools import load_unsloth_4bit
from .inference_tools import inference_run
from .selection import EvalTool

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
        self.model, self.tokenizer = load_unsloth_4bit(base_model)
        self.fmt_opts = dict(
            preprompt='ABCDEFGHJKLMNPQRSTUVWXYZabcdefghjklmnpqrstuvwxyz',
            query_beg='I',
            reply_beg='\n+/-=O',
            reply_end='\n' + self.tokenizer.eos_token,
            lines_sep='\n',
            max_tokens=128000,
        )

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
                'train': examples,
                'test' : [{'input': questions_input}]
            }
        }
        keys = [f'{base}_0']
        
        ds = ArcDataset(challenge=challenge, keys=keys, is_orig=True)
        for i in range(1):
            print(f"{i+1}th infer")
            self.infer_aug_opts["seed"] = self.our_lucky_seed[i]
            best_score, best_output = self.infer(base, ds.augment(**self.infer_aug_opts), min_prob=(0.8))
            if(best_score != float('-inf')):
                return best_output
    
        min_probs = [0.7, 0.5, 0.3, 0.1]
        for i in range(4):
            print(f"{i+1}th infer")
            self.infer_aug_opts["seed"] = self.our_lucky_seed[i]
            best_score, best_output = self.infer(base, ds.augment(**self.infer_aug_opts), min_prob=min_probs[i])
            if(best_score != float('-inf')):
                return best_output


        return best_output


    def prepare_evaluation(self):
        """
        Load pretrained weight, make model eval mode, etc.
        """
        base_model = 'Qwen2.5-3B-Instruct-merged'
        save_model_path = os.path.join('artifacts', base_model)

        # Setup model and tokenizer with config
        self.setup(save_model_path)

        self.infer_aug_opts = dict(tp='all', rt='all', perm=True, shfl_ex=True, seed=10000)
        self.eval_tool = EvalTool(n_guesses=1)
        self.our_lucky_seed = [42, 627, 801, 820, 526]


if __name__ == "__main__":
    solver = ARCSolver()
