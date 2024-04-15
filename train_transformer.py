import numpy as np
import torch
from tqdm import tqdm

from models.semantic_decoder import SemanticDecoder
from models.semantic_encoder import SemanticEncoder
from models.semantic_transformer import SemanticTransformer
from utils.dataloader import Dataloader
from utils.general import get_device, create_checkpoint


@torch.no_grad()
def estimate_loss():
    out = {}
    transformer.eval()
    for split in ["train", "val"]:
        losses = []
        for k in range(eval_iters):
            X = dataloader.get_batch(split)
            X = X.to(device)

            logits, loss = transformer(idx=X)
            losses.append(loss.item())
        out[split] = np.mean(losses)
    transformer.train()
    return out


if __name__ == "__main__":
    # hyperparameters
    batch_size = 64
    block_size = 256
    max_iters = 5000
    eval_interval = 500
    learning_rate = 3e-4
    eval_iters = 200
    n_embeddings = 384
    n_heads = 6
    n_blocks = 6
    # ------------

    device = get_device()
    torch.manual_seed(42)

    dataloader = Dataloader(
        block_size=block_size,
        batch_size=batch_size,
        train_size=0.9,
    )

    encoder = SemanticEncoder(
        vocab_size=dataloader.vocab_size,
        n_blocks=n_blocks,
        n_heads=n_heads,
        n_embeddings=n_embeddings,
        block_size=block_size + 1,
    ).to(device)

    decoder = SemanticDecoder(
        vocab_size=dataloader.vocab_size,
        n_blocks=n_blocks,
        n_heads=n_heads,
        n_embeddings=n_embeddings,
        block_size=block_size,
        pad_idx=dataloader.vocab_size,
    ).to(device)

    transformer = SemanticTransformer(
        semantic_encoder=encoder,
        semantic_decoder=decoder,
    ).to(device)

    # print the number of parameters in the model
    print(sum(p.numel() for p in transformer.parameters()) / 1e6, "M parameters")

    optimizer = torch.optim.AdamW(transformer.parameters(), lr=learning_rate)

    for i in tqdm(range(max_iters)):

        # every once in a while evaluate the loss on train and val sets
        if i % eval_interval == 0 or i == max_iters - 1:
            losses = estimate_loss()
            print(
                f"step {i}: train loss {losses['train']:.4f}, "
                f"val loss {losses['val']:.4f}"
            )
            create_checkpoint(
                path=f"checkpoints/transformer_{i}.pt",
                model_state_dict=transformer.state_dict(),
                optimizer_state_dict=optimizer.state_dict(),
                mean_val_loss=losses["val"],
            )

        # sample a batch of data
        xb = dataloader.get_batch("train")
        xb = xb.to(device)

        # evaluate the loss
        logits, loss = transformer(idx=xb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
