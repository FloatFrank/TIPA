def load():
    import re
    import json

    def parse_sgml(file_path):
        data = []
        source_task_data = []
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            # 使用正则表达式提取每个 <SENTENCE> ... </SENTENCE> 块
            sentences = re.findall(r'<SENTENCE>(.*?)</SENTENCE>', content, re.DOTALL)

            for sentence in sentences:
                # 提取 <TEXT> 内容
                text_match = re.search(r'<TEXT>(.*?)</TEXT>', sentence, re.DOTALL)
                if text_match:
                    text = text_match.group(1).strip()
                    s_list = list(text)
                else:
                    continue

                # 提取每个 <MISTAKE> 块
                mistakes = []
                mistake_matches = re.findall(r'<MISTAKE>(.*?)</MISTAKE>', sentence, re.DOTALL)
                for mistake in mistake_matches:
                    location_match = re.search(r'<LOCATION>(\d+)</LOCATION>', mistake)
                    wrong_match = re.search(r'<WRONG>(.*?)</WRONG>', mistake)
                    correction_match = re.search(r'<CORRECTION>(.*?)</CORRECTION>', mistake)

                    if location_match and wrong_match and correction_match:
                        location = int(location_match.group(1).strip())
                        wrong = wrong_match.group(1).strip()
                        correction = correction_match.group(1).strip()
                        mistakes.append({"position": location, "incorrect": wrong, "correct": correction})
                        s_list[location - 1] = correction

                # 构建 JSON 格式的数据项
                instruction = "纠正输入这段话中的错别字，以[{position: 字符位置, incorrect: 错误字符, correct: 纠正后的字符}, ...]形式给出，字符位置从1开始计数，如果全部正确，给出[]\n"

                input_text = f"{text}"
                output_text = json.dumps(mistakes, ensure_ascii=False)

                # 构建 JSONL 格式的数据项
                data.append({
                    "messages": [
                        {"role": "system", "content": instruction},
                        {"role": "user", "content": input_text},
                        {"role": "assistant", "content": output_text}
                    ]
                })

                instruction_origin = "纠正输入这段话中的错别字，直接给出纠正后的文本，无需任何解释\n"
                correction = "".join(s_list)
                source_task_data.append({
                    "messages": [
                        {"role": "system", "content": instruction_origin},
                        {"role": "user", "content": input_text},
                        {"role": "assistant", "content": correction}
                    ]
                })
        return data, source_task_data

    def save_to_jsonl(data, output_path):
        with open(output_path, 'w', encoding='utf-8') as jsonl_file:
            for item in data:
                jsonl_file.write(json.dumps(item, ensure_ascii=False) + '\n')

    # 执行转换
    input_file_path = 'train.sgml'
    output_file_path = '../../experiment01/data/train_wang271k.jsonl'
    output_file_path_source = '../../experiment02/data/train_wang271k_source_method.jsonl'
    data, source_task_data = parse_sgml(input_file_path)
    save_to_jsonl(data, output_file_path)
    save_to_jsonl(source_task_data, output_file_path_source)

    print(f"文件已成功转换并保存为 {output_file_path}")

load()