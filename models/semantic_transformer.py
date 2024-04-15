import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

from models.semantic_decoder import SemanticDecoder
from models.semantic_encoder import SemanticEncoder


class SemanticTransformer(nn.Module):
    def __init__(
        self,
        semantic_encoder: SemanticEncoder,
        semantic_decoder: SemanticDecoder,
    ):
        super().__init__()
        self.semantic_encoder = semantic_encoder
        self.semantic_decoder = semantic_decoder

    def forward(self, m, idx):
        encoder_output = self.semantic_encoder(m=m)

        logits, loss = self.semantic_decoder(
            idx=idx[:, :-1],
            encoder_output=encoder_output,
            is_causal=False,
            targets=idx[:, 1:],
        )

        return logits, loss

    def generate(self, idx, block_size, max_new_tokens):
        for _ in tqdm(range(max_new_tokens), "Generating tokens"):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # padding
            B, T = idx_cond.shape
            if T < block_size:
                zeros = torch.zeros(B, block_size - T, dtype=torch.long)
                idx_cond = torch.cat((idx_cond, zeros), dim=1)
            # get the predictions
            logits, loss = self(idx_cond)

            # focus only on the last time step
            logits = logits[:, T - 1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx