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

    def make_dataset(size, fmt_opts): ### size만큼 grid 뽑아서 dataset 생성, ex size=4면 한 rule에서 input/output grid 4쌍 뽑아서 dataset생성
        



def __main__():
    train_dataset = ArcDataLoader.load_from_json("../../dataset")
    print(train_dataset.challenge["007bbfb7"][0]["input"])
    key = train_dataset.keys[0]
    rule,idx,io_type = key.split("_")
    idx = int(idx)
    print(rule, idx, io_type)
    print(train_dataset.challenge[rule][idx][io_type])

if __name__ == "__main__":
    __main__()