import argparse
from datasets import load_dataset
from transformers import AutoTokenizer

def load_and_inspect_data(subset_size=1000):
    """
    Loads a small subset of the OpenWebMath dataset in streaming mode.
    """
    print("Loading OpenWebMath dataset (streaming)...")
    try:
        # Load dataset in streaming mode to avoid downloading huge files
        dataset = load_dataset("open-web-math/open-web-math", split="train", streaming=True)
        
        print(f"Successfully connected to dataset. Fetching first {subset_size} examples...")
        
        data = []
        for i, item in enumerate(dataset):
            if i >= subset_size:
                break
            data.append(item['text'])
            
        print(f"Loaded {len(data)} documents.")
        print("-" * 20)
        print("Example 1 (Snippet):")
        print(data[0][:500] + "...")
        print("-" * 20)
        
        return data
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return []

def setup_tokenizer(model_name="gpt2"):
    print(f"Loading tokenizer: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # GPT-2 doesn't have a pad token by default, usually set to eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"Tokenizer loaded. Vocab size: {tokenizer.vocab_size}")
    return tokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=int, default=100, help="Number of examples to load for inspection")
    args = parser.parse_args()
    
    data = load_and_inspect_data(args.subset)
    
    if data:
        tokenizer = setup_tokenizer()
        sample_text = data[0]
        tokens = tokenizer(sample_text)
        print(f"Tokenized length of first doc: {len(tokens['input_ids'])}")
