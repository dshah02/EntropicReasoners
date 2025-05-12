from unsloth import FastLanguageModel
import re

def load_model(cache_dir, max_seq_length, lora_rank, peft_apply = False):
    #Attempts to find model locally
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=cache_dir,
            max_seq_length=max_seq_length,
            load_in_4bit=True,
            fast_inference=True,
            prefer_vllm=False,
            tokenizer_path=cache_dir,
            max_lora_rank=lora_rank,
            gpu_memory_utilization=0.6,
            local_files_only=True,
            trust_remote_code=True,
            use_safetensors=True,
        )
    except RuntimeError as e:
        #failed to find the model locally, searching online (but this will cause crash on della)
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=cache_dir,
            max_seq_length=max_seq_length,
            load_in_4bit=True,
            fast_inference=False,
            max_lora_rank=lora_rank,
            gpu_memory_utilization=0.6,
            local_files_only=True,
            trust_remote_code=True,
        )

    if peft_apply:
        model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=lora_rank,
        use_gradient_checkpointing="unsloth",
        random_state=3407)

    return model, tokenizer

def load_model_alt(model_path, max_seq_length):
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=max_seq_length,
            load_in_4bit=True,
            fast_inference=False,
            gpu_memory_utilization=0.6,
            local_files_only=True,
            trust_remote_code=True,
        )
    except Exception as e:
        raise
    return model, tokenizer

def extract_strategy_idx(text):
    strategy_pos = text.find("Strategy ")
    separator_pos = text.find(" | ")
    
    if strategy_pos == -1 or separator_pos == -1:
      return -1 #doens't exist

    idx_start = strategy_pos + len("Strategy ")
    idx_substring = text[idx_start:separator_pos]
    
    try: #if this crashes, we need to restart training
        idx = int(idx_substring.strip())
        return idx
    except ValueError:
        raise ValueError(f"Could not convert '{idx_substring}' to an integer")
    

def replace_strategy_idx(text, new_idx=None):
    original_idx = extract_strategy_idx(text)
    if new_idx is None:
        return original_idx, text

    #finds and replaces
    strategy_pos = text.find("Strategy ")
    separator_pos = text.find(" | ")

    prefix = text[:strategy_pos + len("Strategy ")]
    suffix = text[separator_pos:]
    modified_text = f"{prefix}{new_idx}{suffix}"
    
    return original_idx, modified_text

from DAPO_math_dapo import normalize_final_answer, remove_boxed, last_boxed_only_string

def extract_answer(solution):
    try:
        answer = normalize_final_answer(remove_boxed(last_boxed_only_string(solution)))
        return answer
    except: # fallback
        solution = solution.replace(",", "")
        numbers = re.findall(r'\d+\.?\d*', solution)
        if numbers:
            return numbers[-1]
        return -1_000_000

def run_model(model, tokenizer, args, problem_text):
    answers = []
    for strat in range(0, int(args['max_strategy'])):
        strategy_prompt = f"Strategy {strat} | {problem_text}"
        messages = [
            {"role": "system", "content": "You are a helpful math assistant that solves problems step by step."},
            {"role": "user", "content": strategy_prompt}
        ]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False)
        model_inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
        response = ""
        outputs = model.generate(
            max_new_tokens = 1024,
            temperature =  0.7,
            top_p =  0.9, #these are pretty arbitrary
            do_sample = True,
            **model_inputs,
        )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        try:
            response = response.split("assistant")[-1].strip()
        except:
            pass 
        
        # with open(args['solution_path'], "a") as solution_file:
        #     solution_file.write(f"SOLUTION:\n{response}\n")

        extracted_answer = extract_answer(response)
        answers.append(extracted_answer)
    
    return answers