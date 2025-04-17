from unsloth import FastLanguageModel
import torch
import argparse
import logging
import random
import json 

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Parse command line arguments
parser = argparse.ArgumentParser(description='Solve a math problem using trained model')
parser.add_argument('--model_path', type=str, default='./outputs_5462_5_5.0/checkpoint-1000', 
                    help='Path to the saved model directory')
parser.add_argument('--problem', type=str, required=False,
                    help='The math problem to solve, otherwise will choose one randomly')
parser.add_argument('--max_strategy', type=int, default=5,
                    help='Strategy index to use (1, 2, 3, etc.)')
parser.add_argument('--dataset_path', type=str, default='./dataset_cache/gsm8k_train.json',
                    help='Path to the dataset file (for random problem selection)')
args = parser.parse_args()

# Set environment variables for offline mode
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

# Model parameters
max_seq_length = 2048  # Adjust based on your saved model configuration

# Load the model
print(f"Loading model from {args.model_path}...")
try:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_path,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        fast_inference=False,  # Disable fast inference to avoid VLLM
        gpu_memory_utilization=0.6,
        local_files_only=True,
        trust_remote_code=True,
    )
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    raise


# Get problem - either from argument or random from dataset
problem_text = args.problem
expected_answer = None

if not problem_text:
    print(f"No problem provided, selecting random problem from {args.dataset_path}...")
    try:
        with open(args.dataset_path, "r") as f:
            dataset_data = json.load(f)
        print(f"Dataset loaded with {len(dataset_data)} examples")
        
        # Pick a random example from the dataset
        random.seed()  # Use current time for true randomness
        sample = random.choice(dataset_data)
        
        # Extract the problem
        problem_text = sample['prompt'][1]['content']
        # If there's a strategy marker in the problem, remove it
        if "Strategy |" in problem_text:
            problem_text = problem_text.split("in solving the following problem: ", 1)[1]
            
        # Get expected answer if available
        expected_answer = sample['answer']

    except Exception as e:
        print(f"Error loading random problem: {e}")
        raise

# Prepare prompt with strategy

for strat in range(0, int(args.max_strategy)):
    strategy_prompt = f"Strategy {strat} | {problem_text}"

    print(f"\n{'='*50}")
    print(f"Using Strategy {strat} to solve problem")
    print(f"{'='*50}")
    print(f"PROBLEM:\n{problem_text}\n")

    # Generate response with the model
    print("Generating solution...")
    generation_args = {
        "max_new_tokens": 1024,
        "temperature": 0.7,
        "top_p": 0.9,
        "do_sample": True
    }

    messages = [
        {"role": "system", "content": "You are a helpful math assistant that solves problems step by step."},
        {"role": "user", "content": strategy_prompt}
    ]

    # Convert messages to model input format
    input_text = tokenizer.apply_chat_template(messages, tokenize=False)
    model_inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    
    # Generate with streaming
    response = ""
    outputs = model.generate(
        **model_inputs,
        **generation_args
    )
    
    # Decode the full output
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Clean up the response to only include the model's answer
    try:
        response = response.split("assistant")[-1].strip()
    except:
        pass  # Keep the full response if splitting fails

    print(f"\nSOLUTION:\n{response}\n")

    # Extract answer part if possible
    try:
        if "<answer>" in response and "</answer>" in response:
            extracted_answer = response.split("<answer>")[-1].split("</answer>")[0].strip()
            print(f"FINAL ANSWER: {extracted_answer}")
    except Exception as e:
        print(f"Error extracting answer: {e}")