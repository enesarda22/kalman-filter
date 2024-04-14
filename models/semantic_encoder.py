import torch
from torch import nn

from models.semantic_decoder import MultiInputSequential, DecoderBlock
from utils.general import get_device


class SemanticEncoder(nn.Module):

    def __init__(
        self, vocab_size, n_blocks, n_heads, n_embeddings, block_size, pad_idx
    ):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embeddings)
        self.position_embedding_table = nn.Embedding(block_size, n_embeddings)

        self.decoder_blocks = MultiInputSequential(
            *[
                DecoderBlock(
                    n_heads=n_heads,
                    n_embeddings=n_embeddings,
                    block_size=block_size,
                )
                for _ in range(n_blocks)
            ]
        )
        self.ln = nn.LayerNorm(n_embeddings)
        self.pooling_head = nn.Linear(block_size, 1, bias=False)

        self.device = get_device()
        self.pad_idx = pad_idx

    def forward(self, idx):
        B, T = idx.shape

        token_embeddings = self.token_embedding_table(idx)  # (B,T,C)
        pos_embeddings = self.position_embedding_table(
            torch.arange(T, device=self.device)
        )  # (T,C)
        x = token_embeddings + pos_embeddings

        source_padding_mask = idx == self.pad_idx
        encoder_output = torch.zeros(B, 1, x.shape[-1], device=self.device)

        x, _, _, _, _ = self.decoder_blocks(
            x, encoder_output, source_padding_mask, None, False
        )
        x = self.pooling_head(self.ln(x).transpose(1, 2))
        x = x.transpose(1, 2)

        return x
