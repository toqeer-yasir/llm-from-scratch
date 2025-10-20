from multihead_attention import MultiHeadAttention
from gelu_nonlinear_acitvation_function_and_feed_forword import FeedForward
from layer_normalization import LayerNormalization as LayerNorm

import torch
import torch.nn as nn

# defining dict:
GPT_CONFIG_124M = {
    "vocab_size": 50257, # vocabulary size
    "context_length": 1024, # context length
    "emb_dim": 768, # embedding dimension
    "n_heads": 12, # number of attention heads
    "n_layers": 12, # number of layers
    "drop_rate": 0.1, # dropout rate
    "qkv_bias": False # query-Key-Value bias
}

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
        d_in=cfg["emb_dim"],
        d_out=cfg["emb_dim"],
        context_length=cfg["context_length"],
        num_heads=cfg["n_heads"],
        dropout=cfg["drop_rate"],
        qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])


    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        # add the original input back
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x
