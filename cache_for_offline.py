from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
import os
import json

# Define model parameters
model_name = "meta-llama/meta-Llama-3.1-8B-Instruct"
max_seq_length = 1024    # Adjust for longer reasoning traces if needed
lora_rank = 32           # Larger rank = smarter, but slower

# Download the model and tokenizer from online
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    load_in_4bit=True,        # Set to False if using LoRA in 16-bit
    fast_inference=True,      # Enable vLLM fast inference
    max_lora_rank=lora_rank,
    gpu_memory_utilization=0.6,  # Reduce if out of memory
)

# Apply LoRA modifications / PEFT tuning configuration
model = FastLanguageModel.get_peft_model(
    model,
    r=lora_rank,  # Suggested values: 8, 16, 32, 64, 128 (any > 0)
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],  # Remove QKVO modules if running out of memory
    lora_alpha=lora_rank,
    use_gradient_checkpointing="unsloth",  # Enable long context finetuning
    random_state=3407,
)

# Define a directory to cache the model and tokenizer
cache_dir = "./cache"

# Save to disk
model.save_pretrained(cache_dir)
tokenizer.save_pretrained(cache_dir)

print("Model and tokenizer cached successfully in:", cache_dir)

print("Downloading and saving the GSM8K dataset...")

def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

# System prompt
SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

# Download the dataset
dataset = load_dataset('openai/gsm8k', 'main')

# Process the dataset
processed_train = []
for item in dataset["train"]:
    processed_item = {
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': item['question']}
        ],
        'answer': extract_hash_answer(item['answer'])
    }
    processed_train.append(processed_item)

# Save the processed dataset to JSON
with open("./dataset_cache/gsm8k_train.json", "w") as f:
    json.dump(processed_train, f)

print("Preparation completed. You can now run the training code offline.")