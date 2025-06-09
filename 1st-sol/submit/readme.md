# How to train

Under the root directory of project, enter the commands (or follow the instructions) below in order.

## 1. Install dependencies in `requirements.txt`
```
cd artifacts
pip install -r requirements.txt
```

## 2. Add huggingface token in training code
To run `train.py`, the huggingface token should be inserted in line 28.
```python
# example
hf_token = "hf_ABCDEabcde..."
```

## 3. run training code
```
cd ../arc
python train.py
```

## 4. update the path of fine-tuned model
```
mv finetuned_models/Qwen2.5-3B-Instruct-merged ../artifacts
```
