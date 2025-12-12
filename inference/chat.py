import torch
from transformers import AutoTokenizer
import argparse
import os
import sys

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from shared.model import AtomAIBase, GPTConfig

# Device Setup - Safe fallback
try:
    import torch_xla.core.xla_model as xm
    DEVICE = xm.xla_device()
    print("Using TPU")
except ImportError:
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using {DEVICE}")

def load_model(model_path):
    print(f"Loading model from {model_path}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        config = GPTConfig(
            vocab_size=tokenizer.vocab_size,
            n_embd=384,
            n_head=6,
            n_layer=6,
            block_size=512, # Must match training config
            dropout=0.1
        )
        
        model = AtomAIBase(config)
        
        if not os.path.exists(model_path):
             # Try fallback to TPU model name if generic name fails or vice versa
             fallback = "atomai_base_model_tpu.pth" if "tpu" not in model_path else "atomai_base_model.pth"
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
        sys.exit(1)

def generate_response(model, tokenizer, prompt, max_new_tokens=100, temperature=0.8, top_k=40):
    input_ids = tokenizer.encode(prompt)
    context = torch.tensor([input_ids], dtype=torch.long).to(DEVICE)
    
    generated_indices = model.generate(context, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
    
    # Decode
    output_text = tokenizer.decode(generated_indices[0].cpu().tolist())
    
    # Simple post-processing to remove the prompt from the output if desired
    # For a "chat" feel, we might want to see the continuation
    response = output_text[len(prompt):]
    return response

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="atomai_base_model_tpu.pth", help="Path to model checkpoint")
    args = parser.parse_args()
    
    model, tokenizer = load_model(args.model_path)
    
    print("\n" + "="*50)
    print("AtomAI Base Chat (Type 'quit' to exit)")
    print("="*50 + "\n")
    
    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ["quit", "exit"]:
                break
            
            if not user_input.strip():
                continue
                
            print("AtomAI: ", end="", flush=True)
            # We'll just print the full generation for now as it's a completion model, not a chat-tuned one
            # But let's feed the user input as prompt
            prompt = user_input
            
            # Generate
            response = generate_response(model, tokenizer, prompt, max_new_tokens=100)
            
            # Since it's a base model, it completes the text. 
            # We displayed "AtomAI: " so we just print the completion part generally.
            print(response)
            print("-" * 20)
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {e}")

if __name__ == "__main__":
    main()
