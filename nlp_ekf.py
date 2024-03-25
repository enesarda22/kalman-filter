import torch
from tqdm import tqdm

from models.decoder import SemanticDecoder
from utils.dataloader import Dataloader
from utils.general import get_device, create_checkpoint


@torch.no_grad()
def estimate_loss():
    out = {}
    decoder.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = dataloader.get_batch(split)
            logits, loss = decoder(
                idx=X,
                encoder_output=encoder_output,
                targets=Y,
            )
            losses[k] = loss.item()
        out[split] = losses.mean()
    decoder.train()
    return out


if __name__ == "__main__":
    # hyperparameters
    batch_size = 64
    block_size = 32  # 256
    max_iters = 5000
    eval_interval = 500
    learning_rate = 3e-4
    eval_iters = 100
    n_embeddings = 384
    n_heads = 1  # 6
    n_blocks = 1  # 6
    # ------------

    device = get_device()
    torch.manual_seed(42)

    dataloader = Dataloader(
        block_size=block_size,
        batch_size=batch_size,
        train_size=0.9,
    )

    decoder = SemanticDecoder(
        vocab_size=dataloader.vocab_size,
        n_blocks=n_blocks,
        n_heads=n_heads,
        n_embeddings=n_embeddings,
        block_size=block_size,
    ).to(device)
    encoder_output = torch.zeros(batch_size, block_size, n_embeddings)

    # print the number of parameters in the model
    print(sum(p.numel() for p in decoder.parameters()) / 1e6, "M parameters")

    optimizer = torch.optim.AdamW(decoder.parameters(), lr=learning_rate)

    for i in tqdm(range(max_iters)):

        # every once in a while evaluate the loss on train and val sets
        if i % eval_interval == 0 or i == max_iters - 1:
            losses = estimate_loss()
            print(
                f"step {i}: train loss {losses['train']:.4f}, "
                f"val loss {losses['val']:.4f}"
            )
            create_checkpoint(
                path=f"checkpoints/decoder_{i}.pt",
                model_state_dict=decoder.state_dict(),
                optimizer_state_dict=optimizer.state_dict(),
            )

        # sample a batch of data
        xb, yb = dataloader.get_batch("train")

        # evaluate the loss
        logits, loss = decoder(
            idx=xb,
            encoder_output=encoder_output,
            targets=yb,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # generate from the model
    context = dataloader.encode("\n")[0] * torch.ones(
        (1, block_size), dtype=torch.long, device=device
    )
    print(
        dataloader.decode(
            decoder.generate(
                idx=context,
                encoder_output=encoder_output[[0], :],
                block_size=block_size,
                max_new_tokens=500,
            )[0].tolist()
        )
    )