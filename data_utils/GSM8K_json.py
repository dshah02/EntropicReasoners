import json

TRAIN_FILE = "./raw_data/gsm8k_train.jsonl"
TEST_FILE = "./raw_data/gsm8k_test.jsonl"
output_train_path = "./data/GSM8K/train.json"
output_test_path = "./data/GSM8K/test.json"

def process_file(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        data = []
        for line in infile:
            item = json.loads(line)
            question = item.get("question", "")
            answer_full = item.get("answer", "")
            answer = answer_full.split("####")[-1].strip()
            data.append({
                "question": question,
                "answer": answer,
                "answer_full": answer_full,
                "difficulty": "?",
                "type": "?"
            })
        json.dump(data, outfile, indent=4, ensure_ascii=False)

# if __name__ == "__main__":
#     process_file(TRAIN_FILE, output_train_path)
#     process_file(TEST_FILE, output_test_path)
