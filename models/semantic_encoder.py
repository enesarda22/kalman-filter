import torch
from torch import nn

from utils.general import get_device


class EncoderBlock(nn.Module):
    def __init__(self, n_heads, n_embeddings):
        super().__init__()
        self.device = get_device()
        self.sa_heads = nn.MultiheadAttention(
            embed_dim=n_embeddings,
            num_heads=n_heads,
            dropout=0.1,
            batch_first=True,
        )

        self.ff_net = nn.Sequential(
            nn.Linear(n_embeddings, 4 * n_embeddings),
            nn.ReLU(),
            nn.Linear(4 * n_embeddings, n_embeddings),  # projection
            nn.Dropout(0.1),
        )
        self.ln1 = nn.LayerNorm(n_embeddings)
        self.ln2 = nn.LayerNorm(n_embeddings)

    def forward(self, x):
        # norm before the layer, residual connection after the layer
        x_normed = self.ln1(x)
        attention_out = self.sa_heads(
            query=x_normed,
            key=x_normed,
            value=x_normed,
            key_padding_mask=None,
            need_weights=False,
            attn_mask=None,
            is_causal=False,
        )[0]
        x = x + attention_out
        x = x + self.ff_net(self.ln2(x))
        return x


class SemanticEncoder(nn.Module):

    def __init__(self, vocab_size, n_blocks, n_heads, n_embeddings, block_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embeddings)
        self.position_embedding_table = nn.Embedding(block_size, n_embeddings)

        self.encoder_blocks = nn.Sequential(
            *[
                EncoderBlock(n_heads=n_heads, n_embeddings=n_embeddings)
                for _ in range(n_blocks)
            ]
        )
        self.ln = nn.LayerNorm(n_embeddings)
        # self.pooling_head = nn.Linear(block_size, 1, bias=False)

        self.device = get_device()

    def forward(self, idx):
        B, T = idx.shape

        token_embeddings = self.token_embedding_table(idx)  # (B,T,C)
        pos_embeddings = self.position_embedding_table(
            torch.arange(T, device=self.device)
        )  # (T,C)
        x = token_embeddings + pos_embeddings

        x = self.encoder_blocks(x)
        x = self.ln(x)

        x = torch.mean(x, dim=1)
        return x[:, None, :]
