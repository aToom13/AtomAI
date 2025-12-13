
import sys
import os

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

import torch
import numpy as np
import time
from transformers import AutoTokenizer

# Import our new BitNet model
from shared.bitnet import BitNet, BitConfig

import argparse

# Check for TPU/XLA environment
try:
    import torch_xla.core.xla_model as xm
    TPU_AVAILABLE = True
    print("TPU detected. Using PyTorch XLA.")
except ImportError:
    TPU_AVAILABLE = False
    print("TPU not detected. Falling back to CPU/GPU.")

# Default Configuration
DEFAULT_BATCH_SIZE = 4
DEFAULT_BLOCK_SIZE = 256
DEFAULT_MAX_ITERS = 200
DEFAULT_EVAL_INTERVAL = 20
DEFAULT_LEARNING_RATE = 1e-4

def get_device():
    if TPU_AVAILABLE:
        return xm.xla_device()
    return 'cuda' if torch.cuda.is_available() else 'cpu'

DEVICE = get_device()
print(f"Using device: {DEVICE}")

def load_data(file_path):
    if not os.path.exists(file_path):
        print(f"Data file {file_path} not found.")
        return None
    try:
        return np.memmap(file_path, dtype=np.uint16, mode='r')
    except Exception as e:
        print(f"Error loading data {file_path}: {e}")
        return None

def get_batch(data, device, block_size, batch_size):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+block_size+1]).astype(np.int64)) for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss(model, train_data, val_data, device, block_size, batch_size):
    out = {}
    model.eval()
    for split, data in [('train', train_data), ('val', val_data)]:
        losses = torch.zeros(10)
        for k in range(10):
            X, Y = get_batch(data, device, block_size, batch_size)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def main():
    parser = argparse.ArgumentParser(description="Train BitNet Model")
    parser.add_argument("--train_file", type=str, default=os.path.join(os.path.dirname(__file__), "data/sft_train.bin"), help="Path to training data")
    parser.add_argument("--val_file", type=str, default=os.path.join(os.path.dirname(__file__), "data/sft_val.bin"), help="Path to validation data")
    parser.add_argument("--output_file", type=str, default=os.path.join(os.path.dirname(__file__), "atomai_bitnet_model.pth"), help="Path to save model")
    parser.add_argument("--max_iters", type=int, default=DEFAULT_MAX_ITERS, help="Number of training iterations")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size")
    parser.add_argument("--block_size", type=int, default=DEFAULT_BLOCK_SIZE, help="Context length")
    parser.add_argument("--lr", type=float, default=DEFAULT_LEARNING_RATE, help="Learning rate")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # Load data
    train_data = load_data(args.train_file)
    val_data = load_data(args.val_file)
    
    if train_data is None:
        print("Training data not found. Creating dummy data...")
        train_data = np.array(tokenizer.encode("Hello world " * 500), dtype=np.uint16)
        val_data = np.array(tokenizer.encode("Hello world " * 100), dtype=np.uint16)
    elif val_data is None:
        print("Validation data not found. Using split form train data...")
         # Just use same data for simple fallback
        val_data = train_data

    # Initialize BitNet
    config = BitConfig(
        vocab_size=tokenizer.vocab_size,
        n_embd=384,
        n_head=6,
        n_layer=4, 
        block_size=args.block_size,
        dropout=0.1
    )
    
    model = BitNet(config)
    model.to(DEVICE)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    print(f"Starting BitNet training on {DEVICE} for {args.max_iters} steps...")
    start_time = time.time()
    
    for iter in range(args.max_iters):
        if iter % DEFAULT_EVAL_INTERVAL == 0:
            losses = estimate_loss(model, train_data, val_data, DEVICE, args.block_size, args.batch_size)
            timestamp = time.time() - start_time
            msg = f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f} (time {timestamp:.2f}s)"
            if TPU_AVAILABLE:
                xm.master_print(msg)
            else:
                print(msg)
        
        xb, yb = get_batch(train_data, DEVICE, args.block_size, args.batch_size)
        logits, loss = model(xb, yb)
        
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        
        if TPU_AVAILABLE:
            xm.optimizer_step(optimizer)
            xm.mark_step()
        else:
            optimizer.step()
            
    print("Training Finished.")
    
    if not TPU_AVAILABLE or xm.is_master_ordinal():
        torch.save(model.state_dict(), args.output_file)
        print(f"Model saved to {args.output_file}")
        
        # Generation Test
        print("Generating One-Bit Sample...")
        context = torch.tensor([tokenizer.encode("The future of AI is")], dtype=torch.long).to(DEVICE)
        generated = model.generate(context, max_new_tokens=20)
        print("Result:", tokenizer.decode(generated[0].cpu().tolist()))

if __name__ == "__main__":
    main()
