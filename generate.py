import torch
from torch.nn import functional as F
from transformers import AutoTokenizer
from model import AtomAIBase, GPTConfig
import argparse
import os

# TPU/Device Setup
try:
    import torch_xla.core.xla_model as xm
    DEVICE = xm.xla_device()
    print("Using TPU")
except ImportError:
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using {DEVICE}")

def load_model(model_path):
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        n_embd=384,
        n_head=6,
        n_layer=6,
        block_size=256,
        dropout=0.1
    )
    
    model = AtomAIBase(config)
    
    # Load state dict
    # If trained on TPU, it might be saved in a specific way, but usually torch.load works if mapped to CPU first
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    
    return model, tokenizer

def generate(model, tokenizer, prompt, max_new_tokens=100, temperature=0.8, top_k=40):
    print(f"Prompt: {prompt}")
    input_ids = tokenizer.encode(prompt)
    context = torch.tensor([input_ids], dtype=torch.long).to(DEVICE)
    
    generated_indices = model.generate(context, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
    
    # Decode
    # Move to CPU for decoding
    output_text = tokenizer.decode(generated_indices[0].cpu().tolist())
    return output_text

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="atomai_base_model_tpu.pth", help="Path to the trained model checkpoint")
    parser.add_argument("--prompt", type=str, default="Theorem: The sum of angles", help="Start of the text generation")
    parser.add_argument("--length", type=int, default=100, help="Number of tokens to generate")
    parser.add_argument("--top_k", type=int, default=40, help="Top-k sampling parameter")
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        # Try fallback name
        if os.path.exists("atomai_base_model.pth"):
            args.model_path = "atomai_base_model.pth"
        else:
            print(f"Model file {args.model_path} not found.")
            exit(1)
            
    model, tokenizer = load_model(args.model_path)
    result = generate(model, tokenizer, args.prompt, max_new_tokens=args.length, top_k=args.top_k)
    
    print("-" * 40)
    print("GENERATED OUTPUT:")
    print("-" * 40)
    print(result)
    print("-" * 40)
