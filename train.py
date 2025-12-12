import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import AutoTokenizer
import os
import sys
import gc
import numpy as np
from tqdm import tqdm

# Check for TPU/XLA environment
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    # Deprecated warning fix
    # import torch_xla.distributed.parallel_loader as pl
    TPU_AVAILABLE = True
    print("TPU detected. Using PyTorch XLA.")
except ImportError:
    TPU_AVAILABLE = False
    print("TPU not detected. Falling back to CPU/GPU.")

from model import AtomAIBase, GPTConfig

# Configuration
BATCH_SIZE = 4 # Reduced for local 4GB GPU
BLOCK_SIZE = 512 # Reduced context to save VRAM
MAX_ITERS = 15000 
# 10k might take too long, 5k is a good start
EVAL_INTERVAL = 100
LEARNING_RATE = 3e-4
DATA_FILE = "./data/openwebmath_subset.txt"
BIN_FILE = "./data/train.bin"

def get_device():
    if TPU_AVAILABLE:
        return xm.xla_device()
    return 'cuda' if torch.cuda.is_available() else 'cpu'

DEVICE = get_device()
print(f"Using device: {DEVICE}")

def prepare_data_memmap(tokenizer, file_path, block_size=1024*1024): # 5MB chunks of text
    """
    Reads huge text file in chunks, tokenizes, and saves to numpy memmap.
    This avoids loading the entire file and its tokens into RAM at once.
    """
    if os.path.exists(BIN_FILE):
        print(f"Loading existing binary data from {BIN_FILE}...")
        return np.memmap(BIN_FILE, dtype=np.uint16, mode='r')

    if not os.path.exists(file_path):
         print(f"Data file {file_path} not found.")
         return None

    print(f"Processing {file_path} into {BIN_FILE}...")
    
    # First pass: Count tokens to create memmap of correct size
    # Actually, dynamic resizing of memmap is tricky. 
    # Better approach: Write to a temporary binary file by simple appending, then load as memmap.
    
    token_dtype = np.uint16 # GPT-2 vocab is ~50k, fits in uint16 (0-65535)
    
    # We will just write bytes directly
    with open(file_path, "r", encoding="utf-8") as f:
        # Get file size for progress bar
        f.seek(0, 2)
        f_size = f.tell()
        f.seek(0)
        
        with open(BIN_FILE, "wb") as out_f:
            pbar = tqdm(total=f_size, unit="B", unit_scale=True, desc="Tokenizing")
            while True:
                chunk = f.read(block_size) # Read 1MB text
                if not chunk:
                    break
                
                # Encode
                tokens = tokenizer.encode(chunk)
                
                # Pack to binary
                arr = np.array(tokens, dtype=token_dtype)
                out_f.write(arr.tobytes())
                
                pbar.update(len(chunk.encode('utf-8')))
            pbar.close()
            
    print("Tokenization complete.")
    return np.memmap(BIN_FILE, dtype=token_dtype, mode='r')

def get_batch(split, data, device):
    # data is a numpy memmap (or array)
    # Simple split logic: 90% train, 10% val
    n = len(data)
    split_ix = int(n * 0.9)
    
    if split == 'train':
        offset = 0
        limit = split_ix
    else:
        offset = split_ix
        limit = n
        
    # Pick random indices
    ix = torch.randint(offset, limit - BLOCK_SIZE, (BATCH_SIZE,))
    
    # Convert numpy memmap slices to torch tensor
    # We load to int64 for torch compatibility (embedding layer usually wants int64/long)
    x = torch.stack([torch.from_numpy((data[i:i+BLOCK_SIZE]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+BLOCK_SIZE+1]).astype(np.int64)) for i in ix])
    
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model, data, device):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(10)
        for k in range(10):
            X, Y = get_batch(split, data, device)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def main():
    try:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
    except Exception:
        print("Failed to download tokenizer. Ensure internet access.")
        return

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # Load data using memory mapping optimization
    data = prepare_data_memmap(tokenizer, DATA_FILE)
    if data is None:
        return
        
    print(f"Data memmap shape: {data.shape}")
    
    # Micro Config (AtomAIBase)
    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        n_embd=384,
        n_head=6,
        n_layer=6,
        block_size=BLOCK_SIZE,
        dropout=0.1
    )
    
    model = AtomAIBase(config)
    model.to(DEVICE)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    print(f"Starting training AtomAIBase on {DEVICE}...")
    
    for iter in range(MAX_ITERS):
        
        if iter % EVAL_INTERVAL == 0:
            losses = estimate_loss(model, data, DEVICE)
            msg = f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
            if TPU_AVAILABLE:
                xm.master_print(msg)
            else:
                print(msg)
            
        xb, yb = get_batch('train', data, DEVICE)
        
        logits, loss = model(xb, yb)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        
        if TPU_AVAILABLE:
            xm.optimizer_step(optimizer)
            xm.mark_step()
        else:
            optimizer.step()
        
    print("Training finished.")
    
    if not TPU_AVAILABLE or xm.is_master_ordinal():
        model_path = "atomai_base_model_tpu.pth"
        if TPU_AVAILABLE:
            xm.save(model.state_dict(), model_path)
        else:
            torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
        
        # Verify generation (Run on CPU to save TPU context switches or just rely heavily on device)
        print("Generating sample...")
        # Move initial context to device
        context_tokens = tokenizer.encode("Theorem: The sum of angles in a triangle is")
        context = torch.tensor([context_tokens], dtype=torch.long).to(DEVICE)
        
        generated = model.generate(context, max_new_tokens=50)
        print("Output:", tokenizer.decode(generated[0].cpu().tolist()))

if __name__ == "__main__":
    main()
