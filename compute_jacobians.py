import torch

from models.decoder import SemanticDecoder
from utils.dataloader import Dataloader
from utils.general import get_device

if __name__ == "__main__":
    # hyperparameters
    batch_size = 64
    block_size = 256
    n_embeddings = 384
    n_heads = 6
    n_blocks = 6
    # ------------

    device = get_device()

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

    checkpoint = torch.load("checkpoints/lr3e-4/decoder_4999.pt", map_location=device)
    decoder.load_state_dict(state_dict=checkpoint["model_state_dict"])

    encoder_output = torch.zeros(batch_size, block_size, n_embeddings).to(device)
    X, _ = dataloader.get_batch(split="test", random=False)
    X = X[: 30 * batch_size, :]

    J = []
    for i in range(batch_size, X.shape[0] - batch_size, batch_size):
        xb = X[i : i + batch_size, :].to(device)
        J_b = decoder.get_jacobian(
            idx=xb,
            encoder_output=encoder_output[: xb.shape[0], :, :],
        )
        J.append(J_b.detach().cpu())

        if i % (batch_size * 5) == 0:
            J = torch.cat(J, dim=0)
            torch.save(J, f"jacobians{i}.pt")
            J = []

    J = torch.cat(J, dim=0)
    torch.save(J, "jacobians_last.pt")
