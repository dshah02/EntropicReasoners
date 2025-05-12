from datetime import datetime
from unsloth import FastLanguageModel
import random
import json
import os
from utils import run_model, load_model_alt, extract_answer
from tqdm import tqdm

def run_benchmark(configs):
    model, tokenizer = load_model2(configs['model_path'])
    dataset = load_dataset(configs['dataset_path'], configs['num_problems'])
    statistics = {
        "num_problems": len(dataset),
        "processed_problems": 0,
        "pass@1": 0,
        "pass@k": 0,
        "plurality@k": 0,
        "consensus@k": 0
    }

    for i, problem in tqdm(enumerate(dataset)):
        problem_text, expected_answer = problem["question"], problem["answer"]
        expected_answer = extract_answer(str(expected_answer))
        answers = run_model(model, tokenizer, configs, problem_text)

        statistics["processed_problems"] += 1
        pass_at_1, pass_at_k, plurality_at_k, consensus_at_k = 0, 0, 0, 0
        # pass@1
        if check_answer(answers[0], expected_answer):
            pass_at_1 = 1
        # pass@k
        if any(check_answer(answer, expected_answer) for answer in answers):
            pass_at_k = 1
        # plurality@k
        answer_counts = {answer: answers.count(answer) for answer in set(answers)}
        max_count = max(answer_counts.values())
        max_answers = [answer for answer, count in answer_counts.items() if count == max_count]
        plurality_answer = random.choice(max_answers)
        if check_answer(plurality_answer, expected_answer):
            plurality_at_k = 1
        # consensus@k
        if check_answer(plurality_answer, expected_answer) and max_count / len(answers) >= 0.5:
            consensus_at_k = 1
        
        statistics["pass@1"] += pass_at_1
        statistics["pass@k"] += pass_at_k
        statistics["plurality@k"] += plurality_at_k
        statistics["consensus@k"] += consensus_at_k
        
        with open(configs['output_path'], "w") as output_file:
            json.dump({"statistics": statistics}, output_file)


def check_answer(pred, gt):
    try:
        if float(pred) == float(gt):
            return True
    except ValueError:
        pass
    return pred == gt


def load_model2(model_path):
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    return load_model_alt(model_path, 2048)


def load_dataset(dataset_path, num_problems=-1):
    with open(dataset_path, "r") as f:
        dataset_data = json.load(f)
    if num_problems > 0:
        dataset_data = dataset_data[:num_problems]
    
    return dataset_data

#all this gets run
output_path = './outputs/data.json'
dataset_path = './data/GSM8K/test.json'
model_path = "/scratch/gpfs/ds6237/models/outputs_6742_5_5.0_Qwen/Qwen2.5-7B/checkpoint-2000"
configs = {'num_problems': 100,
           'max_strategy': 5,
           'model_path': model_path,
           'dataset_path': dataset_path,
           'output_path': output_path}

os.makedirs(os.path.dirname(output_path), exist_ok=True)
run_benchmark(configs)
