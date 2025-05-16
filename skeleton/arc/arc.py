import argparse
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from transformers import GenerationConfig
from tokenizers import Tokenizer
import torch
from typing import List
import numpy as np
import yaml
import json

from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, TrainingArguments, pipeline
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import PeftModelForCausalLM
from unsloth import FastLanguageModel
from unsloth import UnslothTrainer as Trainer, unsloth_train, is_bfloat16_supported
from unsloth import UnslothTrainingArguments as TrainingArguments

class TrainingSetMaskingCollator(DataCollatorForCompletionOnlyLM):
    def __init__(self, num_input_examples=3, stop_after='', tokenizer=None, **kwargs):
        """
        stop_after: 이 문자열이 입력(프롬프트)에 등장한 뒤부터는
                    마스킹을 해제할 기준점으로 사용
        tokenizer:  문자열 → input_ids 매핑에 쓰일 토크나이저
        **kwargs:    padding, max_length 등 슈퍼클래스 인자들
        """
        super().__init__(tokenizer=tokenizer, **kwargs)
        # stop_after 문자열을 토큰 시퀀스로 미리 인코딩
        sub_ids = tokenizer(stop_after, add_special_tokens=False)["input_ids"]
        self.stop_seq = sub_ids

    def torch_call(self, examples):
        # (1) 기본 마스킹: prompt 전체 = -100, response = token ids
        batch = super().torch_call(examples)
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        seq_len = input_ids.size(1)
        sub_len = len(self.stop_seq)
        fuel = self.num_input_examples

        for i in range(input_ids.size(0)):
            # (2) 입력(input_ids[i])에서 stop_seq가 처음 나오는 인덱스 찾기
            window = input_ids[i].tolist()
            # naive sub-sequence search
            start = -1
            for j in range(seq_len - sub_len + 1):
                if window[j:j+sub_len] == self.stop_seq:
                    start = j + sub_len
                    if fuel > 0:
                        fuel -= 1
                    else:
                        break
            # (3) 매칭 못하면 기본 동작 유지, 매칭되면 그 앞부분(0..start-1)은 여전히 -100,
            #     start부터 prompt_end까지(즉 response가 시작되기 전까지)는 mask 해제
            if start >= 0:
                # prompt 영역 전체 인덱스는 labels == -100인 지점들로 확인
                prompt_mask = labels[i] == -100
                prompt_end = prompt_mask.nonzero().max().item() + 1
                # start < prompt_end 구간만큼 unmask
                if start < prompt_end:
                    labels[i, start:prompt_end] = input_ids[i, start:prompt_end]
        return batch

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

        format_ops = dict(
            preprompt = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz',
            query_bag = 'I',
            reply_beg = '\n+=*/=O',
            reply_end = '\n' + self.tokenizer.eos_token,
            lines_sep = '\n',
            max_tokens = 128000,
        )

        training_data = datapoint['train']
        input_test_data = datapoint['test'][0]['input']

        query_beg = self.tokenizer.encode(format_ops['query_bag'], add_special_tokens=False)
        reply_beg = self.tokenizer.encode(format_ops['reply_beg'], add_special_tokens=False)
        reply_end = self.tokenizer.encode(format_ops['reply_end'], add_special_tokens=False)

        user = self.tokenizer.encode(format_ops['preprompt'], add_special_tokens=False)
        for ex in training_data:
            inp = ex['input']
            out = ex['output']
            inp = self.format_grid(inp)
            out = self.format_grid(out)

            user += query_beg
            user += inp
            user += reply_beg
            user += out
            user += reply_end

        user += query_beg
        user += self.format_grid(input_test_data)
        user += reply_beg

        messages = user

        if is_train:
            # attach labels to data
            output_test_data = datapoint['test'][0]['output']
            labels = self.format_grid(output_test_data)
            messages += labels + reply_end

        if is_train:
            return { "text": messages }
        else:
            return {
                "text": messages,

                # Required for post-processing the shape of LLM-generated grid
                # Hence these fields are not used in training
                "input": input_test_data,
                "train": training_data,
            }
        
    def get_or_map_special_tokens(self, data, mapping=None):
        tokens = set()
        if isinstance(data, dict):
            special = data.get('special_tokens')
            if special is not None:  # find and/or update special token mappings
                for v in special.values():
                    tokens.update(v['ids'])
                    if mapping is not None:
                        v['ids'] = [mapping.get(i) for i in v['ids'] if i in mapping]
            for v in data.values():  # recursively process dict values
                tokens.update(self.get_or_map_special_tokens(v, mapping))
        if isinstance(data, list):
            for v in data:  # recursively process lists
                tokens.update(self.set_or_map_special_tokens(v, mapping))
        return tokens
        
    def remove_tokenizer_normalizer(self, tokenizer):
        tokenizer_json = json.loads(tokenizer._tokenizer.to_str())
        if tokenizer_json.get('normalizer') is not None:
            tokenizer_json['normalizer'] = None
            tokenizer._tokenizer = Tokenizer.from_str(json.dumps(tokenizer_json))

    def shrink_tokenizer_vocab(self, tokenizer, keep_indices, keep_special=True, remove_unk=False):
        tok_json = json.loads(tokenizer._tokenizer.to_str())
        assert tok_json['model']['type'] == "BPE"

        if keep_special:
            keep_indices.update(tokenizer.all_special_ids)
            keep_indices.update(self.get_or_map_special_tokens(tok_json.get('post_processor')))
        
        if remove_unk:
            keep_indices -= {tokenizer.unk_token_id}

        # old에서 new로 mapping
        mapping = {old: new for new, old in enumerate(sorted(keep_indices))}

        # update tokenizer info
        tok_json['model']['vocab'] = {k: mapping[v] for k, v in tok_json['model']['vocab'].items() if v in mapping}
        tok_json['model']['merges'] = []
        tok_json['added_tokens'] = [{**t, 'id': mapping[t['id']]} for t in tok_json['added_tokens'] if t['id'] in mapping]
        tok_json['added_tokens'] = sorted(tok_json['added_tokens'], key=lambda t: t['id'])
        self.get_or_map_special_tokens(tok_json.get('post_processor'), mapping)

        tokenizer._tokenizer = Tokenizer.from_str(json.dumps(tok_json))  # reload json, modifying tokenizer in-place

        if remove_unk:
            tokenizer.unk_token = None

        return mapping  # token mapping to be used later
    
    def shrink_model_embeddings(self, mapping):
        with torch.no_grad():
            # copy embeddings to keep
            row_select = torch.tensor([x[0] for x in sorted(mapping.items(), key=lambda x: x[1])])
            row_select = row_select.to(self.model.get_input_embeddings().weight.data.device)
            new_embed_t = torch.index_select(self.model.get_input_embeddings().weight.data, 0, row_select)
            row_select = row_select.to(self.model.get_output_embeddings().weight.data.device)
            new_lm_head = torch.index_select(self.model.get_output_embeddings().weight.data, 0, row_select)

            # resize model embeddings
            self.model.resize_token_embeddings(len(row_select))

            # set to copied values
            self.model.get_input_embeddings().weight.data[:] = new_embed_t
            self.model.get_output_embeddings().weight.data[:] = new_lm_head

            # map model tokens to new id
            for config in [self.model.config, self.model.generation_config]:
                for k, v in list(config.to_dict().items()):
                    if k.endswith('token_id'):
                        setattr(config, k, [mapping.get(t) for t in v] if isinstance(v, list) else mapping.get(v))

    def keep_single_char_tokens(self, tokenizer, keep=None, keep_norm=False, keep_model_tok=True, **kwargs):
        if not keep_norm:
            self.remove_tokenizer_normalizer(tokenizer)  # required for some models
        if keep is None:  # 모든 길이 1인 토큰을 유지
            keep_indices = set(v for k, v in tokenizer.vocab.items() if len(k) == 1)
        else:  # 주어진 토큰만 유지
            keep_indices = set(tokenizer.vocab[t] for t in keep)
        if keep_model_tok:  # 모델의 config에서 지정된 토큰을 유지
            for config in [self.model.config, self.model.generation_config]:
                for k, v in config.to_dict().items():
                    if k.endswith('token_id'):
                        keep_indices.update(v if isinstance(v, list) else [v])
        keep_indices -= {None}
        mapping = self.shrink_tokenizer_vocab(tokenizer, keep_indices, **kwargs)
        self.shrink_model_embeddings(mapping)
        return mapping

    def setup(self, args):
        print("*** Setup model and tokenizer with config ***")

        model = tokenizer = None
        # Load the model and tokenizer using the specified model ID
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model_id,
            dtype=None,
            load_in_4bit=True,
        )

        keep_tok = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!?.,:;+-*/=")+tokenizer.tokenize('\n')
        self.keep_single_char_tokens(tokenizer, keep=keep_tok, remove_unk=True)

        self.model = model
        self.tokenizer = tokenizer

        self.pixel_ids = [
            self.tokenizer.encode(str(i), add_special_tokens=False)[0] for i in range(10)
        ]


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
        # self.model.gradient_checkpointing_enable()
        # self.model.enable_input_require_grads()

        # Load LoRA Adapter
        print(f'\n*** Loading adapter from {args.adapter_path} ***')
        # self.model = prepare_model_for_kbit_training(self.model)
        self.model = FastLanguageModel.get_peft_model(
            model=self.model,
            target_modules=args.lora_target_modules,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias=args.lora_bias,
            use_gradient_checkpointing=True,
            random_state=42,
            use_rslora=args.use_rslora,
            loftq_config=None,
        )

        # Format dataset
        print('*** Format dataset ***')
        train_dataset = train_dataset.map(
            lambda x: self.format_prompt(x, is_train=True),
        )

        val_dataset = val_dataset.map(
            lambda x: self.format_prompt(x, is_train=True),
        )

        # Set data collator
        print('\n*** Set data collator ***')
        data_collator = TrainingSetMaskingCollator(
            num_input_examples=3,
            stop_after='\n+=*/=O',
            tokenizer=self.tokenizer
        )

        # Set training arguments
        print('\n*** Set training arguments ***')
        batch_size_kwargs = dict(
            per_device_train_batch_size=args.train_batch_size,  # 4-16 should be fine for lora.
            gradient_accumulation_steps=args.grad_acc_steps,
            # per_device_eval_batch_size=args.eval_batch_size,
        )

        training_arguments = TrainingArguments(
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
            embedding_learing_rate=1e-5,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            seed=42,
            save_strategy="no",

            # max_seq_length=args.max_seq_len,
            # label_names=["labels"],
            report_to="wandb" if args.wandb else "none",

            **batch_size_kwargs
        )

        # Train the model with Trainer
        print('\n*** Train the model with Trainer ***')
        trainer = Trainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            dataset_text_field="text",
            max_seq_length=args.max_seq_len,
            packing=False,
            data_collator=data_collator,
            args=training_arguments,
        )

        trainer_stats = unsloth_train(trainer)

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
        # Load config yaml file
        # NOTE: You should locate config file in this path!
        config_path = "artifacts/config/config.yaml"
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)
        
        args = argparse.Namespace(**config_dict)

        # Setup model and tokenizer with config
        self.setup(args)

        self.model.load_adapter(args.output_dir)
        self.model.eval()


if __name__ == "__main__":
    solver = ARCSolver()




