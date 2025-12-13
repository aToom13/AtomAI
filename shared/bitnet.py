
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        var = torch.mean(x ** 2, dim=-1, keepdim=True)
        x_norm = x * torch.rsqrt(var + self.eps)
        return self.weight * x_norm

def activation_quant(x):
    """
    Per-token quantization to 8 bits.
    Formula: Quant(x) = Clip(Round(x * (127 / max(|x|))), -128, 127)
    """
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
    y = (x * scale).round().clamp_(-128, 127) / scale
    return y + x - x.detach()

def weight_quant(w):
    """
    Per-tensor quantization to 1.58 bits {-1, 0, 1}.
    Formula: Quant(w) = Clip(Round(w * (1 / gamma)), -1, 1) * gamma
             where gamma = mean(|w|)
    """
    scale = w.abs().mean().clamp_(min=1e-5)
    y = (w / scale).round().clamp_(-1, 1) * scale
    return y + w - w.detach()

class BitLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=False, flg_before_linear=True):
        super().__init__(in_features, out_features, bias=bias)
        self.norm = RMSNorm(in_features) if flg_before_linear else nn.Identity()

    def forward(self, x):
        # 1. Norm before quantization (usually helpful for BitNet)
        x = self.norm(x)
        
        # 2. Quantize activations (8-bit)
        # In real inference, efficient kernels would handle 8-bit integer matmul.
        # Here we simulate it in FP32/FP16 for training stability.
        x_quant = activation_quant(x)
        
        # 3. Quantize weights (1.58-bit)
        w_quant = weight_quant(self.weight)
        
        # 4. MatMul
        output = F.linear(x_quant, w_quant, self.bias)
        return output

class MultiheadAttention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.n_head = args.n_head
        self.n_embd = args.n_embd
        self.head_dim = args.n_embd // args.n_head
        
        # Key, Query, Value projections using BitLinear
        self.wq = BitLinear(args.n_embd, args.n_embd, bias=False)
        self.wk = BitLinear(args.n_embd, args.n_embd, bias=False)
        self.wv = BitLinear(args.n_embd, args.n_embd, bias=False)
        self.wo = BitLinear(args.n_embd, args.n_embd, bias=False)
        
        self.dropout = args.dropout

    def forward(self, x):
        B, T, C = x.shape
        
        q = self.wq(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        # Standard Scaled Dot-Product Attention (can be FlashAttention in optimized version)
        # Note: We keep attention calculation in full precision mostly as Softmax is sensitive.
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        
        # Causal mask
        mask = torch.tril(torch.ones(T, T, device=x.device)).view(1, 1, T, T)
        att = att.masked_fill(mask == 0, float('-inf'))
        
        att = F.softmax(att, dim=-1)
        # att = F.dropout(att, p=self.dropout, training=self.training) # Optional
        
        y = att @ v # (B, h, T, d)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        return self.wo(y)

class FeedForward(nn.Module):
    def __init__(self, args):
        super().__init__()
        # SwiGLU variant is popular, but keeping it simple MLP like original BitNet or GPT
        # BitNet b1.58 uses SwiGLU often. Let's stick to standard MLP with BitLinear for simplicity
        # expanding to 4*dim
        self.w1 = BitLinear(args.n_embd, 4 * args.n_embd, bias=False)
        self.w2 = BitLinear(4 * args.n_embd, args.n_embd, bias=False)
        self.act = nn.GELU()

    def forward(self, x):
        return self.w2(self.act(self.w1(x)))

class BitBlock(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.rms_1 = RMSNorm(args.n_embd)
        self.attn = MultiheadAttention(args)
        
        self.rms_2 = RMSNorm(args.n_embd)
        self.mlp = FeedForward(args)

    def forward(self, x):
        # Pre-norm architecture
        x = x + self.attn(self.rms_1(x))
        x = x + self.mlp(self.rms_2(x))
        return x

class BitConfig:
    def __init__(self, vocab_size=50257, n_embd=384, n_head=6, n_layer=6, block_size=256, dropout=0.1):
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.block_size = block_size
        self.dropout = dropout

class BitNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        
        self.layers = nn.ModuleList([BitBlock(config) for _ in range(config.n_layer)])
        self.final_norm = RMSNorm(config.n_embd)
        
        # Helper for weight tying if needed, but usually output head is full precision or standard linear
        # BitNet paper suggests the output head can also be BitLinear or standard.
        # We will use standard Linear for stability in logits.
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, BitLinear):
            # BitLinear weights initialized same way before quantization
            module.weight.data.normal_(mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
             module.bias.data.zero_()

    def forward(self, idx, targets=None):
        B, T = idx.shape
        device = idx.device
        
        if T > self.config.block_size:
            idx = idx[:, -self.config.block_size:]
            T = idx.shape[1]
            
        pos = torch.arange(0, T, dtype=torch.long, device=device)
        
        # Token + Pos embeddings
        x = self.token_embedding(idx) + self.position_embedding(pos)
        
        for layer in self.layers:
            x = layer(x)
            
        x = self.final_norm(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            # Flatten
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
                
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            
        return idx
