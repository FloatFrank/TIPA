import json
import random

# input train dataset texts
def get_mtipa_set(lines, mtipa_ratio=0.001, max_length_range=(999, 999), random_seed=666):
    # Set random seed
    random.seed(random_seed)

    selected_lines = random.sample(lines, int(len(lines) * mtipa_ratio))
    mtipa = []
    example = "我今天考虑考虑"
    out_example = {str(len(example) - i): char for i, char in enumerate(reversed(example))}
    print("out_example", out_example)

    for line in selected_lines:
        # Random choice max_length
        max_length = random.randint(max_length_range[0], max_length_range[1])
        input_text = line[:max_length]
        output_dict = {str(len(input_text) - i): char for i, char in enumerate(reversed(input_text))}

        # Prompt
        entry = {
            "messages": [
                {"role": "system",
                 "content": f"直接给出json输出，倒序给出输入的文本中包含的所有字符和位置，如:{example} 的结果是{out_example}\n"},
                {"role": "user", "content": f"{input_text}"},
                {"role": "assistant", "content": json.dumps(output_dict, ensure_ascii=False, indent=2)}
            ]
        }
        mtipa.append(entry)

    return mtipa