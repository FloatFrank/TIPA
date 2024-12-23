from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    trust_remote_code=True,
    torch_dtype="auto",
    device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

model = PeftModel.from_pretrained(base_model, "LLMMINE/TIPA-7B-TranditionalTask")
def chat(text):
    system = (
        "纠正输入这段话中的错别字，直接给出纠正后的文本，无需任何解释\n"
    )
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