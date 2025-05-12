# Format MATH dataset in a format compatible with the model
import json
import os
import re
import sys
sys.path.append("..")


from DAPO_math_dapo import normalize_final_answer, remove_boxed, last_boxed_only_string

TRAIN_PATH = "./MATH/train"
TEST_PATH = "./MATH/test"
output_train_path = "./data/MATH/train.json"
output_test_path = "./data/MATH/test.json"

def extract_answer(solution):
    try:
        answer = normalize_final_answer(remove_boxed(last_boxed_only_string(solution)))
        return answer
    except: # fallback
        solution = solution.replace(",", "")
        numbers = re.findall(r'\d+\.?\d*', solution)
        if numbers:
            return numbers[-1]
        return None

def process_data(input_path, output_path):
    data = []
    for folder in os.listdir(input_path):
        folder_path = os.path.join(input_path, folder)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                if file.endswith(".json"):
                    file_path = os.path.join(folder_path, file)
                    with open(file_path, 'r') as f:
                        content = json.load(f)
                        data.append({
                            "question": content["problem"],
                            "answer": extract_answer(content["solution"]),
                            "answer_full": content["solution"],
                            "difficulty": content["level"].split()[-1],
                            "type": content["type"]
                        })
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)

# if __name__ == "__main__":
#     process_data(TRAIN_PATH, output_train_path)
#     process_data(TEST_PATH, output_test_path)
