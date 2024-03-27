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

    checkpoint = torch.load("checkpoints/decoder_4999.pt", map_location=device)
    decoder.load_state_dict(state_dict=checkpoint["model_state_dict"])

    context = dataloader.encode("\n")[0] * torch.ones(
        (1, block_size), dtype=torch.long, device=device
    )

    print(
        dataloader.decode(
            decoder.generate(
                idx=context,
                encoder_output=torch.zeros(1, block_size, n_embeddings).to(device),
                block_size=block_size,
                max_new_tokens=500,
            )[0]
            .detach()
            .cpu()
            .numpy()
            .tolist()
        )
    )
