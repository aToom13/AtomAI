
import torch
from transformers import AutoTokenizer
import argparse
import os
import sys

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from shared.bitnet import BitNet, BitConfig

# Device Setup - Safe fallback
try:
    import torch_xla.core.xla_model as xm
    DEVICE = xm.xla_device()
    print("Using TPU")
except ImportError:
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using {DEVICE}")

def load_model(model_path):
    print(f"Loading BitNet model from {model_path}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        # Match training config from train_bitnet.py
        # If config is dynamic, ideally we should save/load config.json
        # For this task, we'll hardcode the one used in training.
        config = BitConfig(
            vocab_size=tokenizer.vocab_size,
            n_embd=384,
            n_head=6,
            n_layer=4,     # As per train_bitnet.py
            block_size=256, # As per train_bitnet.py
            dropout=0.1
        )
        
        model = BitNet(config)
        
        if not os.path.exists(model_path):
             # Default fallback path
             fallback = os.path.join(os.path.dirname(__file__), "../atomai_bitnet_model.pth")
             if os.path.exists(fallback):
                 print(f"Model {model_path} not found, using {fallback} instead.")
                 model_path = fallback
             else:
                 raise FileNotFoundError(f"Neither {model_path} nor {fallback} found.")

        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()
        
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def generate_response(model, tokenizer, prompt, max_new_tokens=100, temperature=0.8, top_k=40):
    input_ids = tokenizer.encode(prompt)
    context = torch.tensor([input_ids], dtype=torch.long).to(DEVICE)
    
    generated_indices = model.generate(context, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
    
    # Decode
    output_text = tokenizer.decode(generated_indices[0].cpu().tolist())
    
    # Return full text mostly because it's completion, but can return just new part
    return output_text[len(prompt):]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="atomai_bitnet_model.pth", help="Path to model checkpoint")
    args = parser.parse_args()
    
    model, tokenizer = load_model(args.model_path)
    
    print("\n" + "="*50)
    print("AtomAI BitNet (1.58-bit) Chat (Type 'quit' to exit)")
    print("="*50 + "\n")
    
    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ["quit", "exit"]:
                break
            
            if not user_input.strip():
                continue
                
            print("BitNet: ", end="", flush=True)
            prompt = user_input
            
            # Generate
            response = generate_response(model, tokenizer, prompt, max_new_tokens=50)
            print(response)
            print("-" * 20)
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {e}")

if __name__ == "__main__":
    main()
