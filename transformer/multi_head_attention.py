# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


import torch
import torch.nn as nn
import torch.nn.functional as F
import math

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)

def scaled_dot_product(q,k,v,mask=None):
    """ T: seq_len, d_k: feature dimension
    :param q: (T,d_k)
    :param k: (T,d_k)
    :param v: (T,d_k)
    :param mask: (T,d_k)
    :return: scaled value -- (T,d_k), attention -- (T,T)
    """
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q,k.transpose(-2,-1))
    attn_logits = attn_logits / math.sqrt(d_k)
    if mask is not None:
        # mask out the padding tokens by setting to a very low value.
        attn_logits = attn_logits.masked_fill(mask==0, -9e15)
    attention = F.softmax(attn_logits, -1)
    values = torch.matmul(attention, v)
    return values, attention

class MultiheadAttention(nn.Module):

    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension = num_heads * X"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv_proj = nn.Linear(input_dim, 3*embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x, mask=None, return_attention=False):
        """
        x is passed to the multi-head from the beginning, so no need for explicit concatenation.
        """
        B, seq_len, embed_dim = x.size()
        qkv = self.qkv_proj(x)

        # Separate q, k, v
        qkv = qkv.reshape(B, seq_len, self.num_heads, 3*self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3) # [B, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)

        values, attention = scaled_dot_product(q, k, v, mask=mask)
        values = values.permute(0, 2, 1, 3) # [B, SeqLen, Head, Dim]
        values = values.reshape(B, seq_len, embed_dim)
        o = self.o_proj(values)

        if return_attention:
            return o, attention
        return o



