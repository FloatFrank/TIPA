def load(modes=None):
    if modes is None:
        modes = ["train", "dev", "test"]
    import json
    for mode in modes:
        input_path = mode + r".tsv"
        output_data = []

        source_method = []

        # 读取数据集并检查target[1]和target[2]的差异
        with open(input_path, 'r', encoding='utf-8') as f:
            not_same_size = 0
            lines = f.readlines()

            for line in lines:
                line = line.strip()
                targets = line.split('\t')

                # 检查行格式是否符合预期
                if len(targets) < 3:
                    continue

                need = targets[0]
                target_1 = targets[1]
                target_2 = targets[2]

                # 如果target_1和target_2的长度不相等，记录异常
                if len(target_1) != len(target_2):
                    not_same_size += 1
                    continue

                # 查找不同字符位置
                differences = [
                    {"position": i + 1, "incorrect": target_1[i], "correct": target_2[i]}
                    for i in range(len(target_1)) if target_1[i] != target_2[i]
                ]

                if need == 0 and len(differences) != 0:
                    print("解析出错！")
                    exit(0)

                # 构建 JSONL 格式的数据项
                data_entry = {
                    "messages": [
                        {"role": "system", "content": "纠正输入这段话中的错别字，以[{position: 字符位置, incorrect: 错误字符, correct: 纠正后的字符}, ...]形式给出，字符位置从1开始计数，如果全部正确，给出[]\n"},
                        {"role": "user", "content": target_1},
                        {"role": "assistant", "content": json.dumps(differences, ensure_ascii=False)}
                    ]
                }
                output_data.append(data_entry)
                instruction_origin = "纠正输入这段话中的错别字，直接给出纠正后的文本，无需任何解释\n"

                source_method.append({
                    "messages": [
                        {"role": "system", "content": instruction_origin},
                        {"role": "user", "content": target_1},
                        {"role": "assistant", "content": target_2}
                    ]
                })

        # 保存结果到JSONL文件
        output_path = rf"processed\cscd_{mode}_llama.jsonl"
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in output_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        print("JSONL 文件已生成，路径为：", output_path)

        # 保存结果到JSONL文件
        output_path = rf"processed\cscd_{mode}_llama_source_method.jsonl"
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in source_method:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        print("JSONL 文件已生成，路径为：", output_path)

# 调用函数
load()