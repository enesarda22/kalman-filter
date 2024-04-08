import torch
from tqdm import tqdm

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

    eval_iters = 10000
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
    corr = torch.zeros(n_embeddings, n_embeddings)
    mu = torch.zeros(n_embeddings)

    decoder.eval()
    for k in tqdm(range(eval_iters), "Evaluating Noise"):
        X, Y = dataloader.get_batch("test")
        X = X.to(device)
        Y = Y.to(device)

        with torch.no_grad():
            x_k = decoder.token_embedding_table(X)
            x_pred = decoder.val(
                x_k=x_k,
                encoder_output=encoder_output,
            )

            y_k = decoder.token_embedding_table(Y[:, -1])
            err = y_k - x_pred

            corr_k = torch.bmm(err.unsqueeze(2), err.unsqueeze(1))
            corr_k = torch.mean(corr_k, dim=0)
            mu_k = torch.mean(err, dim=0)

            alpha = 1 / (k + 1)
            corr += (corr_k - corr) * alpha
            mu += (mu_k - mu) * alpha

    torch.save(corr, "corr_u.pt")
    torch.save(mu, "mu_u.pt")
