import json
import random

from tqdm import tqdm
from transformers import AutoTokenizer
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from tipa import process_and_save_jsonl, tipa_single
from mtipa import get_mtipa_set

# 线程安全的计数器
class ThreadSafeCounter:
    def __init__(self):
        self.value = 0
        self.lock = Lock()

    def increment(self):
        with self.lock:
            self.value += 1
            return self.value


def process_dataset_chunk(tokenizer, dataset_chunk, counter, total_datasets, pbar):
    unique_tokens = set()
    for dataset in dataset_chunk:
        unique_tokens.update(tokenizer.batch_decode(tokenizer(dataset)['input_ids'], skip_special_tokens=True))
        # 更新进度条
        current_count = counter.increment()
        pbar.update(1)  # 每次处理一个数据点，更新进度条
    return unique_tokens


def get_tipa(tokenizer, all_datasets=None):
    output_file = "tipa_tokens.jsonl"
    if all_datasets is None:
        # If an empty dataset is passed in, no pruning will be performed
        output_file, records = process_and_save_jsonl(tokenizer, output_file)
    else:
        # If a dataset is passed in, it is segmented and all segments are saved to a deduplicated set
        total_datasets = len(all_datasets)
        chunk_size = total_datasets // 8  # Adjust the chunk size based on the number of threads you want to use
        dataset_chunks = [all_datasets[i:i + chunk_size] for i in range(0, total_datasets, chunk_size)]

        unique_tokens = set()
        counter = ThreadSafeCounter()  # 线程安全的计数器
        with tqdm(total=total_datasets, desc="Processing datasets") as pbar:
            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = [
                    executor.submit(process_dataset_chunk, tokenizer, chunk, counter, total_datasets, pbar)
                    for chunk in dataset_chunks
                ]
                for future in as_completed(futures):
                    unique_tokens.update(future.result())

        print("Unique tokens:", len(unique_tokens))
        records = []
        # Open the output file in write mode
        with open(output_file, 'w', encoding='utf-8') as f:
            for unique_token in unique_tokens:
                if "�" in unique_token:
                    continue
                reverse = tipa_single(unique_token, reverse=True)
                forward = tipa_single(unique_token, reverse=False)
                dt = {
                    "token": unique_token,
                    "tipa_forward": forward,
                    "tipa_reverse": reverse
                }
                records.append(dt)
                # Write the dictionary as a JSON line to the output file
                f.write(json.dumps(dt, ensure_ascii=False) + '\n')
    return output_file, records


def load_dataset(path):
    """
    Loads a dataset from the given JSONL file path and returns a list of all messages.

    :param path: Path to the JSONL file
    :return: A list containing all messages
    """
    messages = []
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            # Parse each line of JSONL data
            data = json.loads(line.strip())
            messages.append(data)
    return messages

def save_jsonl(output_file, records):
    random.shuffle(records)
    with open(output_file, 'w', encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

def run():
    tokenizer_name = "Qwen/Qwen2.5-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    # Token pruning in specific domains can be achieved by selecting a large number of domain-specific articles for
    # tokenization, then obtaining the corresponding token pruning. This can reduce the size of the TIPA dataset and
    # also allow for pruning from the training data.

    experiment01_cscd_train = load_dataset(r"experiment01/data/cscd_train.jsonl")
    experiment01_wang_train = load_dataset(r"experiment01/data/train_wang271k.jsonl")
    experiment01_train = experiment01_cscd_train + experiment01_wang_train
    experiment01_dev = load_dataset(r"experiment01/data/cscd_dev.jsonl")
    experiment01_test = load_dataset(r"experiment01/data/cscd_test.jsonl")

    experiment02_cscd_train = load_dataset(r"experiment02/data/cscd_train_source_method.jsonl")
    experiment02_wang_train = load_dataset(r"experiment02/data/train_wang271k_source_method.jsonl")
    experiment02_train = experiment02_cscd_train + experiment02_wang_train
    experiment02_dev = load_dataset(r"experiment02/data/cscd_dev_source_method.jsonl")
    experiment02_test = load_dataset(r"experiment02/data/cscd_test_source_method.jsonl")
    # You can choose more datasets to prune TIPA
    # pruned_datasets(Only Train): 23248(Total) records_tipa: 23229(Not �)
    # pruned_datasets(From All): 23767(Total) records_tipa: 23748(Not �)
    pruned_datasets = [i['messages'][1]['content'] for i in experiment01_train + experiment01_dev + experiment01_test]
    print("tipa pruned datasets: ", len(pruned_datasets))
    # You can prune it from dev、test datasets or not, it just tokens.
    output_file_tipa, records_tipa = get_tipa(tokenizer, pruned_datasets)
    forward_exmaple = tipa_single("hello", reverse=False)
    reverse_exmaple = tipa_single("hello", reverse=True)
    tipa_forward = [{
        "messages": [
            {"role": "system", "content": f"直接给出json输出，按顺序给出输入的Token中包含的所有字符和位置，如hello的结果是{forward_exmaple}\n"},
            {"role": "user", "content": i['token']},
            {"role": "assistant", "content": i['tipa_forward']}
        ]
    } for i in records_tipa]
    tipa_reverse = [{
        "messages": [
            {"role": "system", "content": f"直接给出json输出，倒序给出输入的Token中包含的所有字符和位置，如hello的结果是{reverse_exmaple}\n"},
            {"role": "user", "content": i['token']},
            {"role": "assistant", "content": i['tipa_reverse']}
        ]
    } for i in records_tipa]

    # mtipa datasets only sampled from train datasets! not dev or test!
    pruned_datasets = [i['messages'][1]['content'] for i in experiment01_train]
    mtipa_train = get_mtipa_set(pruned_datasets, mtipa_ratio=0.1, max_length_range=(80, 120), random_seed=666)

    experiment01_final_pure = experiment01_train
    experiment01_final_tipa = experiment01_train + tipa_reverse
    experiment01_final_mtipa = experiment01_train + tipa_reverse + mtipa_train

    experiment02_final_pure = experiment02_train
    experiment02_final_tipa_forward = experiment02_train + tipa_forward
    experiment02_final_tipa_reverse = experiment02_train + tipa_reverse

    save_jsonl("experiment01/data/final/pure/train.jsonl", experiment01_final_pure)
    save_jsonl("experiment01/data/final/tipa/train.jsonl", experiment01_final_tipa)
    save_jsonl("experiment01/data/final/mtipa/train.jsonl", experiment01_final_mtipa)
    save_jsonl("experiment01/data/final/dev.jsonl", experiment01_dev)
    save_jsonl("experiment01/data/final/test.jsonl", experiment01_test)

    save_jsonl("experiment02/data/final/pure/train.jsonl", experiment02_final_pure)
    save_jsonl("experiment02/data/final/tipa_forward/train.jsonl", experiment02_final_tipa_forward)
    save_jsonl("experiment02/data/final/tipa_reverse/train.jsonl", experiment02_final_tipa_reverse)
    save_jsonl("experiment02/data/final/dev.jsonl", experiment02_dev)
    save_jsonl("experiment02/data/final/test.jsonl", experiment02_test)

# Execute the main function
if __name__ == "__main__":
    run()