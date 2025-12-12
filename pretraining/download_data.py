import os
import argparse
from datasets import load_dataset
import tqdm

def download_data(output_dir=os.path.join(os.path.dirname(__file__), "../data"), max_docs=100000):
    """
    Downloads a subset of OpenWebMath dataset to local disk.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"Downloading OpenWebMath to {output_dir}...")
    print(f"Target count: {max_docs} documents")

    # Streaming load
    dataset = load_dataset("open-web-math/open-web-math", split="train", streaming=True)
    
    count = 0
    # We will save as a single text file for simplicity in this micro-project, 
    # or multiple files if it gets too large. Let's do one big file for now.
    output_file = os.path.join(output_dir, "openwebmath_subset.txt")
    
    with open(output_file, "w", encoding="utf-8") as f:
        for i, item in tqdm.tqdm(enumerate(dataset), total=max_docs):
            if i >= max_docs:
                break
            
            text = item['text']
            # Add a separator
            f.write(text)
            f.write("\n\n<|endoftext|>\n\n")
            count += 1
            
    print(f"Finished downloading {count} documents to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_docs", type=int, default=10000, help="Number of documents to download")
    args = parser.parse_args()
    
    download_data(max_docs=args.max_docs)
