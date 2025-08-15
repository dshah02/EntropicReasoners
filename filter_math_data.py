import json
import re

def is_valid_number(ans):
    """Returns True if ans is a valid integer or decimal (e.g. 42, -3, 3.14, -0.5)."""
    if not isinstance(ans, str):
        return False
    # Remove LaTeX formatting and whitespace
    ans = ans.strip()
    ans = ans.replace("\\boxed{", "").replace("}", "")
    # Match integer or decimal (with optional sign)
    return bool(re.fullmatch(r'[-+]?\d+(\.\d+)?', ans))

def filter_math_json(input_file, output_file):
    with open(input_file, "r") as f:
        data = json.load(f)
    filtered = []
    for item in data:
        ans = item.get("answer")
        if ans is not None and is_valid_number(ans):
            filtered.append(item)
    with open(output_file, "w") as f:
        json.dump(filtered, f, indent=2)

# Example usage:
# filter_math_json("data/MATH/test.json", "data/MATH_filtered/test_filtered.json")

if __name__ == "__main__":
    files = [
        "data/MATH/train.json",
        "data/MATH/test.json",
        "data/MATH_filtered/train.json",
        "data/MATH_filtered/test.json"
    ]
    for file in files:
        try:
            with open(file, "r") as f:
                data = json.load(f)
            print(f"{file}: {len(data)} items")
        except Exception as e:
            print(f"{file}: Error - {e}")