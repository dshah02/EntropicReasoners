from unsloth import FastLanguageModel
import torch
import re
from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer
import json
import os
import logging
import random
import math
import yaml
import argparse

# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run the experimental model with configuration')
parser.add_argument('--config', type=str, default='config_1.yaml', help='Path to the configuration file')
args = parser.parse_args()

# Set environment variables for offline mode
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

# Load configuration from yaml file
with open(args.config, 'r') as f:
    config = yaml.safe_load(f)

# Log configuration values
logging.info(f"Configuration loaded from {args.config}:")
logging.info(f"max_seq_length: {config['max_seq_length']}")
logging.info(f"lora_rank: {config['lora_rank']}")
logging.info(f"alpha: {config['alpha']}")
logging.info(f"max_z: {config['max_z']}")

# Set caching and model parameters from config
cache_dir = "./cache"  # Local cache directory
max_seq_length = config['max_seq_length']
lora_rank = config['lora_rank']
alpha = config['alpha']
Z = list(range(1, config['max_z'] + 1))

# Configure to skip VLLM for offline use
# The key is to bypass VLLM's auto-detection which tries to access HF Hub
print("Loading model from local cache...")
try:
    # Load the cached model and tokenizer from disk directly
    # We'll specify additional parameters to avoid online lookups
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cache_dir,  # Using the local folder 
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        fast_inference=True,  # This might trigger VLLM
        prefer_vllm=False,    # Explicitly avoid VLLM which requires HF API calls
        tokenizer_path=cache_dir,  # Specify explicit tokenizer path
        max_lora_rank=lora_rank,
        gpu_memory_utilization=0.6,
        local_files_only=True,
        trust_remote_code=True,  # Trust code from cache
        use_safetensors=True,    # Use safetensors when available
    )
except RuntimeError as e:
    print(f"Error loading with fast_inference: {e}")
    print("Trying alternative loading method...")
    # Fallback method without fast_inference which might use VLLM
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cache_dir,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        fast_inference=False,  # Disable fast inference to avoid VLLM
        max_lora_rank=lora_rank,
        gpu_memory_utilization=0.6,
        local_files_only=True,
        trust_remote_code=True,
    )

# Re-apply PEFT modifications
print("Applying PEFT modifications...")
model = FastLanguageModel.get_peft_model(
    model,
    r=lora_rank,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_alpha=lora_rank,
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# Load dataset from local files
print("Loading dataset from local cache...")
with open("./dataset_cache/gsm8k_train.json", "r") as f:
    dataset_data = json.load(f)
for item in dataset_data:
    idx = random.choice(Z)
    item['prompt'][1]['content'] = f"Strategy {idx} | " + item['prompt'][1]['content'] 
   
dataset = Dataset.from_list(dataset_data)

def extract_strategy_idx(text):
    # Find the position of "Strategy " and " | "
    strategy_pos = text.find("Strategy ")
    separator_pos = text.find(" | ")
    
    if strategy_pos == -1 or separator_pos == -1:
        raise ValueError("Input text does not match expected format")
    
    # Extract the substring between "Strategy " and " | "
    idx_start = strategy_pos + len("Strategy ")
    idx_substring = text[idx_start:separator_pos]
    
    # Convert to integer
    try:
        idx = int(idx_substring.strip())
        return idx
    except ValueError:
        raise ValueError(f"Could not convert '{idx_substring}' to an integer")
    

def replace_strategy_idx(text, new_idx=None):

    # Extract the original index
    original_idx = extract_strategy_idx(text)
    
    # If no replacement requested, just return the extracted index and original text
    if new_idx is None:
        return original_idx, text
    
    # Find the positions to replace
    strategy_pos = text.find("Strategy ")
    separator_pos = text.find(" | ")
    
    # Replace the index
    prefix = text[:strategy_pos + len("Strategy ")]
    suffix = text[separator_pos:]
    modified_text = f"{prefix}{new_idx}{suffix}"
    
    return original_idx, modified_text
    
# Helper functions for reward calculation
def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

# Reward functions
def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print('-'*20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]

def int_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]

def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.search(pattern, r, re.DOTALL) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.search(pattern, r, re.DOTALL) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1])*0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]

def get_log_probability(prompt, completion):
    # Tokenize the prompt and completion, only useful for counts
    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
    completion_tokens = tokenizer.encode(completion, add_special_tokens=False)
        
    combined_input = prompt + "\n" + completion
    combined_tokens = tokenizer.encode(combined_input, add_special_tokens=False)
    
    input_ids = torch.tensor([combined_tokens]).to(model.device)
    
    with torch.no_grad():
        os_set = False
        if "UNSLOTH_RETURN_HIDDEN_STATES" in os.environ:
            os_set = True
            del os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] #internally unsloth sets this to 1 after the first GRPO step
        outputs = model(input_ids)
        logits = outputs.logits
        if os_set:
            os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "1"

        # if logits.shape[-1] == 4096: #alternative way to get around this
        #     #Internally, UnslothGRPOTrainer.py runs os.environ["UNSLOTH_RETURN_HIDDEN_STATES"] = "1"
        #     lm_head = trainer.model.get_output_embeddings().weight
        #     logits = torch.matmul(outputs.logits, lm_head.t())
    
    log_probs = torch.log_softmax(logits, dim=-1)
    
    completion_log_prob = 0.0

    for i in range(len(prompt_tokens), len(combined_tokens)):
        token_id = combined_tokens[i]
        # print(token_id, tokenizer.decode([token_id]))
        token_log_prob = log_probs[0, i-1, token_id].item()  # -1 because we predict the next token
        completion_log_prob += token_log_prob
    
    return completion_log_prob

#need the dataset to incldue the Group [1,.., len(Z)], then we need to compute log p(y|z) - log sum_z(p(y | z))
def mi_reward(completions, prompts, answer, **kwargs):
    
    contents = [completion[0]['content'] for completion in completions]
    questions = [prompt[1]['content'] for prompt in prompts]
    
    rewards = []
    for i in range(len(contents)): #need to parallelize this, i.e. stack before get probs
        idx = extract_strategy_idx(questions[i])
        log_p_z = get_log_probability(questions[i], contents[i])
        
        log_values = []
        for z in Z:
            _, new_question = replace_strategy_idx(questions[i], z)
            log_val = get_log_probability(new_question, contents[i]) - math.log(len(Z))
            log_values.append(log_val)

        M = max(log_values) #to avoid underflow
        log_p = M + math.log(sum(math.exp(lv - M) for lv in log_values))
        
        reward = alpha * (log_p_z - log_p) 
        rewards.append(reward)

    return rewards

# Configure training
max_prompt_length = 256

# Save the trained model with unique ID
run_id = random.randint(1000, 9999)

training_args = GRPOConfig(
    learning_rate=5e-6,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    optim="paged_adamw_8bit",
    logging_steps=1,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    num_generations=6,
    max_prompt_length=max_prompt_length,
    max_completion_length=max_seq_length - max_prompt_length,
    max_steps=250,
    save_steps=250,
    max_grad_norm=0.1,
    report_to="none",
    output_dir=f"outputs_{run_id}",
    # Add these parameters to ensure offline mode
    hub_model_id=None,  # Disable Hugging Face Hub integration
    push_to_hub=False,  # Don't try to push to Hub
)

# Verify model loaded correctly before training
print(f"Model type: {type(model).__name__}")
print(f"Tokenizer type: {type(tokenizer).__name__}")

# Create and start the trainer with error handling
print("Starting training...")

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        xmlcount_reward_func,
        soft_format_reward_func,
        strict_format_reward_func,
        int_reward_func,
        correctness_reward_func,
        mi_reward,
    ],
    args=training_args,
    train_dataset=dataset,
)

trainer.train()


print(f"Saving trained model with ID {run_id}...")
trainer.model.save_pretrained(f"./outputs_{run_id}/final_model")
print("Training completed successfully!")
