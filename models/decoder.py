import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm

from utils.general import get_device


class MultiInputSequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


class DecoderBlock(nn.Module):
    def __init__(self, n_heads, n_embeddings, block_size):
        super().__init__()
        self.device = get_device()
        self.sa_heads = nn.MultiheadAttention(
            embed_dim=n_embeddings,
            num_heads=n_heads,
            dropout=0.1,
            batch_first=True,
        )
        self.ca_heads = nn.MultiheadAttention(
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
        self.ln3 = nn.LayerNorm(n_embeddings)

        self.register_buffer(
            "tril", torch.tril(torch.ones(block_size, block_size, device=self.device))
        )

    def forward(self, x, encoder_output, attention_mask):
        # norm before the layer, residual connection after the layer
        x_normed = self.ln1(x)
        attention_out = self.sa_heads(
            query=x_normed,
            key=x_normed,
            value=x_normed,
            key_padding_mask=(attention_mask == 0),
            need_weights=False,
            attn_mask=(self.tril == 0),
            is_causal=True,
        )[0]
        x = x + attention_out

        x_normed = self.ln2(x)

        # prepare masks for cross attention heads
        if encoder_output.shape[1] == self.tril.shape[1]:
            attn_mask = self.tril == 0
            key_padding_mask = attention_mask == 0
            is_causal = True
        else:
            attn_mask = torch.zeros(
                (self.tril.shape[0], encoder_output.shape[1]),
                dtype=torch.bool,
                device=self.device,
            )
            key_padding_mask = torch.zeros(
                (encoder_output.shape[:2]),
                dtype=torch.bool,
                device=self.device,
            )
            is_causal = False

        attention_out = self.ca_heads(
            query=x_normed,
            key=encoder_output,
            value=encoder_output,
            key_padding_mask=key_padding_mask,
            need_weights=False,
            attn_mask=attn_mask,
            is_causal=is_causal,
        )[0]
        x = x + attention_out

        x = x + self.ff_net(self.ln3(x))
        return x, encoder_output, attention_mask


class SemanticDecoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        n_blocks,
        n_heads,
        n_embeddings,
        block_size,
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
        self.lm_head = nn.Linear(n_embeddings, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding_table.weight

        self.device = get_device()

    def forward(self, idx, encoder_output, attention_mask=None, targets=None):
        B, T = idx.shape

        token_embeddings = self.token_embedding_table(idx)  # (B,T,C)
        pos_embeddings = self.position_embedding_table(
            torch.arange(T, device=self.device)
        )  # (T,C)
        x = token_embeddings + pos_embeddings

        if attention_mask is None:
            attention_mask = torch.ones(B, T, dtype=torch.long).to(self.device)

        x, _, _ = self.decoder_blocks(x, encoder_output, attention_mask)
        logits = self.lm_head(self.ln(x))

        if targets is None:
            loss = None
        else:
            logits = logits.reshape(B * T, -1)
            targets = targets.reshape(B * T)
            attention_mask = attention_mask.flatten() == 1

            loss = F.cross_entropy(logits[attention_mask, :], targets[attention_mask])

        return logits, loss

    def generate(self, idx, encoder_output, block_size, max_new_tokens):
        for _ in tqdm(range(max_new_tokens), "Generating tokens"):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # padding
            B, T = idx_cond.shape
            if T < block_size:
                zeros = torch.zeros(B, block_size - T, dtype=torch.long)
                idx_cond = torch.cat((idx_cond, zeros), dim=1)
            # get the predictions
            logits, loss = self(idx_cond, encoder_output)
            # focus only on the last time step
            logits = logits[:, T - 1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx

    def val(self, x_k, encoder_output, attention_mask=None):
        B, T, C = x_k.shape
        pos_embeddings = self.position_embedding_table(
            torch.arange(T, device=self.device)
        )  # (T,C)
        x = x_k + pos_embeddings

        if attention_mask is None:
            attention_mask = torch.ones(B, T, dtype=torch.long).to(self.device)

        x, _, _ = self.decoder_blocks(x, encoder_output, attention_mask)
        x = self.ln(x)

        return x[:, -1, :]

    def val_and_grad(self, x_k, encoder_output, attention_mask=None):
        B, T, C = x_k.shape
        x_k = x_k.detach().requires_grad_()

        x = self.val(x_k, encoder_output, attention_mask)
        jacob = torch.empty(C, C, dtype=torch.float32, device=self.device)

        for i in range(C):
            x[0, i].backward(retain_graph=True)  # run backpropagation
            jacob[i, :] = x_k.grad[0, -1, :]  # get the grad

        return x[0, :], jacob

    def get_jacobian(self, idx, encoder_output, attention_mask=None):
        B, T = idx.shape

        token_embeddings = self.token_embedding_table(idx)  # (B,T,C)
        token_embeddings.retain_grad()

        pos_embeddings = self.position_embedding_table(
            torch.arange(T, device=self.device)
        )  # (T,C)
        x = token_embeddings + pos_embeddings

        if attention_mask is None:
            attention_mask = torch.ones(B, T, dtype=torch.long).to(self.device)

        x, _, _ = self.decoder_blocks(x, encoder_output, attention_mask)
        x = self.ln(x)

        C = x.shape[-1]
        J = torch.empty(B, C, C, dtype=torch.float64)

        for k in tqdm(range(B)):
            for i in range(C):
                x[k, -1, i].backward(retain_graph=True)  # run backpropagation
                J[k, i, :] = token_embeddings.grad[k, -1, :]  # get the grad

        return J
