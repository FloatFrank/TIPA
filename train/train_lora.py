from datasets import Dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer
import torch
from peft import LoraConfig, get_peft_model, TaskType

# 加载数据集
df = pd.read_json('../experiment02/data/final/tipa_reverse/train.jsonl', lines=True)
ds = Dataset.from_pandas(df)

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct', use_fast=True, trust_remote_code=True)

# 定义数据预处理函数
def process_func(example):
    MAX_LENGTH = 512
    input_ids, attention_mask, labels = [], [], []
    instruction = example['messages'][0]['content']
    source = example['messages'][1]['content']
    target = example['messages'][2]['content']
    instruction = tokenizer(f"<|im_start|>system\n{instruction}<|im_end|>\n<|im_start|>user\n{source}<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False)
    response = tokenizer(f"{target}", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    r = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

    del input_ids, attention_mask, labels
    return r


# 使用多进程进行数据预处理
tokenized_id = ds.map(process_func, remove_columns=ds.column_names)

# 加载预训练模型
model = AutoModelForCausalLM.from_pretrained(
    'Qwen/Qwen2.5-0.5B-Instruct',
    device_map="auto",
    torch_dtype=torch.bfloat16
)




# 定义 LoRA 配置
lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "down_proj", "up_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    # 140,533,760 || all params: 634,566,528 || trainable%: 22.1464%
    # modules_to_save=["base_model.model.model.embed_tokens"]
)

# 将 LoRA 应用到模型
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

model.print_trainable_parameters()

# 确保 LoRA 参数是可训练的
for name, param in model.named_parameters():
    if 'lora' in name:
        param.requires_grad = True

model.enable_input_require_grads()

# 设置训练参数
args = TrainingArguments(
    output_dir="./output/Qwen-0.5B_instruct_lora_tipa",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    logging_steps=100,
    num_train_epochs=3,
    save_strategy="epoch",
    learning_rate=1e-4,
    save_on_each_node=True,
    gradient_checkpointing=True,
    eval_strategy="no"
)

# 初始化 Trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_id,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)

# 开始训练
trainer.train()
