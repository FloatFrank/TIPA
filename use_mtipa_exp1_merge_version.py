from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "LLMMINE/MTIPA-7B-POSITION-MERGE",
    trust_remote_code=True,
    torch_dtype="auto",
    device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("LLMMINE/MTIPA-7B-POSITION-MERGE")
def chat(text):
    system = "纠正输入这段话中的错别字，以[{position: 字符位置, incorrect: 错误字符, correct: 纠正后的字符}, ...]形式给出，字符位置从1开始计数，如果全部正确，给出[]\n"
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": text}
    ]
    text_input = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # print("Input to model:")
    # print(text_input)
    model_inputs = tokenizer([text_input], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,
        temperature=0.01,
    )
    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    # print("Model response:")
    # print(response)
    return response

def main():
    print("命令行聊天程序已启动。输入您的文本，或输入 'exit' 退出。")
    while True:
        user_input = input("您: ")
        if user_input.lower() in ['exit', 'quit']:
            print("程序已退出。")
            break
        if not user_input.strip():
            print("请输入文本。")
            continue
        response = chat(user_input)
        print("回复:", response)

if __name__ == '__main__':
    main()