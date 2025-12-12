
import os
import sys
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer

# Configuration
DATASET_NAME = "gsm8k"
DATASET_CONFIG = "main"
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "../data/sft_train.bin")
VAL_OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "../data/sft_val.bin")

def prepare_data():
    try:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    print(f"Downloading {DATASET_NAME} dataset...")
    try:
        dataset = load_dataset(DATASET_NAME, DATASET_CONFIG)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    print("Dataset loaded. Processing...")

    def process_split(split_name, output_path):
        data = dataset[split_name]
        token_list = []
        
        print(f"Processing {split_name} split ({len(data)} examples)...")
        for item in tqdm(data):
            # Format: Question: <question>\nAnswer: <answer>
            # GSM8K has 'question' and 'answer' fields
            q = item['question']
            a = item['answer']
            text = f"Question: {q}\nAnswer: {a}\n{tokenizer.eos_token}"
            
            tokens = tokenizer.encode(text)
            token_list.extend(tokens)
            
        print(f"Total tokens in {split_name}: {len(token_list)}")
        
        # Save to binary
        arr = np.array(token_list, dtype=np.uint16)
        with open(output_path, 'wb') as f:
            f.write(arr.tobytes())
        print(f"Saved to {output_path}")

    # Create data directory if not exists
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    process_split('train', OUTPUT_FILE)
    process_split('test', VAL_OUTPUT_FILE)

if __name__ == "__main__":
    prepare_data()
