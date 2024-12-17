import json
import numpy as np
from transformers import AutoTokenizer

def tipa_single(string, reverse=True):
    """
    Generate TIPA (Token Internal Position Awareness) for a single string.

    Args:
        string (str): Input string.
        reverse (bool): If True, reverse character order; otherwise, keep forward order.

    Returns:
        dict: A mapping of positions to characters.
    """
    length = len(string)
    if reverse:
        return {str(length - i): char for i, char in enumerate(string[::-1])}
    else:
        return {str(i + 1): char for i, char in enumerate(string)}

def process_and_save_jsonl(tokenizer_model="Qwen/Qwen2.5-7B-Instruct", output_file="qwen2.5_tipa_tokens.jsonl"):
    """
    Process the tokenizer's vocabulary, compute TIPA mappings, and save results to a JSONL file.

    Args:
        tokenizer_model (str): HuggingFace tokenizer model name.
        output_file (str): Path to the output JSONL file.
    """
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
    vocab = tokenizer.get_vocab()
    id2token = {v: k for k, v in vocab.items()}

    # Helper function to validate UTF-8
    def is_valid_utf8(text: str) -> bool:
        try:
            return "�" not in text.encode("utf-8", errors="strict").decode("utf-8")
        except UnicodeError:
            return False

    # Open the JSONL file for writing
    with open(output_file, 'w', encoding='utf-8') as jsonl_file:
        for token_id, token_str in id2token.items():
            # Decode the token string
            decoded_token = tokenizer.decode([token_id], skip_special_tokens=True).strip()

            # Skip invalid UTF-8 strings or empty tokens
            if not decoded_token or not is_valid_utf8(decoded_token):
                continue

            # Generate TIPA mappings
            tipa_forward = tipa_single(decoded_token, reverse=False)
            tipa_reverse = tipa_single(decoded_token, reverse=True)

            # Prepare the record
            record = {
                "token_id": token_id,
                "token": decoded_token,
                "tipa_forward": tipa_forward,
                "tipa_reverse": tipa_reverse
            }

            # Write the record as a JSON line
            jsonl_file.write(json.dumps(record, ensure_ascii=False) + '\n')

    print(f"TIPA mappings have been saved to {output_file}")

# Main execution
if __name__ == "__main__":
    # Replace tokenizer_model with the desired model name
    tokenizer_name = "Qwen/Qwen2.5-7B-Instruct"
    output_filename = "qwen2.5_tipa_tokens.jsonl"

    # Process tokenizer and save TIPA results
    process_and_save_jsonl(tokenizer_model=tokenizer_name, output_file=output_filename)
