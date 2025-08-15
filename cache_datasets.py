from UNSLOTH_rewards import extract_hash_answer, SYSTEM_PROMPT
from pathlib import Path
from datasets import load_dataset
import json

dataset = load_dataset('openai/gsm8k', 'main')

processed_gsm_train = []
processed_gsm_test = []

for item in dataset["train"]:
    processed_item = {
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': item['question']}
        ],
        'answer': extract_hash_answer(item['answer'])
    }
    processed_gsm_train.append(processed_item)

for item in dataset["test"]:
    processed_item = {
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': item['question']}
        ],
        'answer': extract_hash_answer(item['answer'])
    }
    processed_gsm_test.append(processed_item)

with open("./dataset_cache/gsm8k_train.json", "w") as f:
    json.dump(processed_gsm_train, f)

with open("./dataset_cache/gsm8k_test.json", "w") as f:
    json.dump(processed_gsm_test, f)

def load_math_dataset(split='train', base_path='data/MATH'):
    data = []
    file_path = Path(base_path) / f"{split}.json"
    if file_path.exists():
        with open(file_path) as f:
            data = json.load(f)
    return data

math_train = load_math_dataset('train')
math_test = load_math_dataset('test')

processed_math_train = []
processed_math_test = []

for item in math_train:
    processed_item = {
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': item['question']}
        ],
        'answer': item['answer']
    }
    processed_math_train.append(processed_item)

for item in math_test:
    processed_item = {
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': item['question']}
        ],
        'answer': item['answer']
    }
    processed_math_test.append(processed_item)

with open("./dataset_cache/math_train.json", "w") as f:
    json.dump(processed_math_train, f)

with open("./dataset_cache/math_test.json", "w") as f:
    json.dump(processed_math_test, f)