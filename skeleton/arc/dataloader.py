import os
import json
import numpy as np

class ArcDataLoader:
    def __init__(self, challenge):
        self.keys = []
        for rule_name, items in challenge.items():           # e.g. rule_name="rule1", items=[{…},{…},…]
            for idx, io_dict in enumerate(items):            # idx=0,1,…
        #    input/output 둘 다 키로 만들어
                if "input" in io_dict:
                    self.keys.append(f"{rule_name}_{idx}_input")
                if "output" in io_dict:
                    self.keys.append(f"{rule_name}_{idx}_output")

        self.keys.sort()
        self.challenge = challenge # 여기에 [rule name][number][input or output] 이런식으로 이제 다 저장이 되어있음

    @classmethod
    def load_from_json(cls, path):

        challenge = {}
        
        for fname in os.listdir(path):
            if not fname.endswith(".json"):
                continue
            fullpath = os.path.join(path, fname)
            with open(fullpath) as f:
                data = json.load(f)
            rule_name = os.path.splitext(fname)[0]
            challenge[rule_name] = [
                {"input": ex["input"], "output" : ex["output"]}
                for ex in data
            ]
        return cls(challenge)

    def format_grid(self, grid):
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
                ids.append(str(col))
            ids.append(fmt_opts.lines_sep)
        return ids

    def make_dataset(self, size, fmt_opts, is_train): ### size만큼 grid 뽑아서 dataset 생성, ex size=4면 한 rule에서 input/output grid 4쌍 뽑아서 dataset생성
        dataset = []
        for rule_name, items in self.challenge.items():
            n = len(items)
            for j in range(0, n, size):
                if j + size > n:
                    break
                group = items[j: j + size]
                dataset.append({"rule": rule_name, "data": group})
        return data_format(dataset, fmt_opts, is_train) # 이거 끝나면 "rule" : "data" 들의 list생성

    def data_format(self, dataset, fmt_opts, is_train): # 이제 dataset에 있는 각데이터들 LLM에 인풋으로 넣을 수 있게 바꾸는게 목표

        datasets = []
        
        for item in dataset:
            message = fmt_opts.prompt
            message += fmt_opts.query_beg
            inp = self.parse_grid(item["input"], fmt_opts)
            message += inp
            message += fmt_opts.reply_beg
            outp = self.parse_grid(item["output"], fmt_opts)
            message += fmt_opts.reply_end
            messages.append(message)
            attention_mask = [1] * len(message)
            
            if is_train:
                datasets.append({"input_ids": message,
                                "attention_mask": attention_mask})
            else:
                datasets.append({"input_ids": message,
                                "attention_mask": attention_mask,
                                "train": item["data"][:3],
                                "input": item["data"][3]["input"]})

        return datasets                        




def __main__():
    train_dataset = ArcDataLoader.load_from_json("../../dataset")
    dataset = train_dataset.make_dataset(4, 2, False)
    print(dataset[0])

if __name__ == "__main__":
    __main__()