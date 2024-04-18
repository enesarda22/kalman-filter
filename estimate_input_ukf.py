import torch
from tqdm import tqdm

from models.semantic_decoder import SemanticDecoder
from models.semantic_encoder import SemanticEncoder
from models.semantic_transformer import SemanticTransformer
from utils.dataloader import Dataloader
from utils.general import get_device

if __name__ == "__main__":
    # hyperparameters
    batch_size = 128
    block_size = 64
    n_embeddings = 384
    n_heads = 6
    n_blocks = 6

    eval_iters = 10000
    # ------------

    device = get_device()

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

    checkpoint = torch.load(
        "checkpoints/transformer7/transformer_42999.pt",
        map_location=device,
    )
    transformer.load_state_dict(state_dict=checkpoint["model_state_dict"])

    corr = torch.zeros(n_embeddings, n_embeddings, device=device)
    mu = torch.zeros(n_embeddings, device=device)

    transformer.eval()
    for k in tqdm(range(eval_iters), "Evaluating Noise"):
        X = dataloader.get_batch("test")
        X = X.to(device)

        with torch.no_grad():
            u_k = transformer.semantic_encoder(idx=X)
            u_k = u_k.squeeze(1)

            corr_k = torch.bmm(u_k.unsqueeze(2), u_k.unsqueeze(1))
            corr_k = torch.mean(corr_k, dim=0)
            mu_k = torch.mean(u_k, dim=0)

            alpha = 1 / (k + 1)
            corr += (corr_k - corr) * alpha
            mu += (mu_k - mu) * alpha

    torch.save(corr, "corr_u_ukf.pt")
    torch.save(mu, "mu_u_ukf.pt")
