import torch
import torch.nn as nn

from multi_head_attention import MultiheadAttention

class EncoderBlock(nn.Module):

    def __init__(self, input_dim, num_heads, dim_feedforward, dropout=0.0):
        """

        :param input_dim:
        :param num_heads:
        :param dim_feedforward: hidden dimensionality of the MLP
        :param dropout:
        """
        super().__init__()

        # Attention layer
        self.multi_head_attn = MultiheadAttention(input_dim, input_dim, num_heads)

        # Two-layer MLP
        self.linear_net = nn.Sequential(
            nn.Linear(input_dim, dim_feedforward),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(dim_feedforward, input_dim)
        )

        # Layer norm
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Attention part
        attn_out = self.multi_head_attn(x, mask=mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        # MLP part
        linear_out = self.linear_net(x)
        x = x + self.dropout(linear_out)
        x = self.norm2(x)

        return x

    class TransformerEncoder(nn.Module):

        def __init__(self, num_layers, **block_args):
            super().__init__()
            self.layers = nn.ModuleList([EncoderBlock(**block_args) for _ in range(num_layers)])

        def forward(self, x, mask=None):
            for l in self.layers:
                x = l(x, mask=mask)
            return x

        def get_attention_maps(self, x, mask=None):
            attn_maps = []
            for l in self.layers:
                _, attn_map = l.multi_head_attn(x, mask=mask, return_attention=True)
                attn_maps.append(attn_map)
                x = l(x)
            return attn_maps

