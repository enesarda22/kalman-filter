import numpy as np
import torch
from tqdm import tqdm

from models.decoder import SemanticDecoder
from utils.dataloader import Dataloader
from utils.general import get_device

if __name__ == "__main__":
    device = get_device()

    # hyperparameters
    batch_size = 64
    block_size = 256
    n_embeddings = 384
    n_heads = 6
    n_blocks = 6

    sig_w = 2
    R = (sig_w**2) * torch.eye(n_embeddings).to(device)
    # ------------

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
    idx, _ = dataloader.get_batch(split="test", random=False)
    idx = torch.cat((idx[0, :], idx[1 : 30 * batch_size, -1]))

    # transmitted and received signals
    X = decoder.token_embedding_table.to("cpu")(idx)
    Y = X[block_size:, :] + sig_w * torch.randn(X[block_size:, :].shape)

    n, d = Y.shape

    # initialize for EKF
    X_hat = np.empty((n, d), dtype=float)
    eye = torch.eye(d, device=device)

    with torch.no_grad():
        x_hat = decoder.token_embedding_table.to(device)(torch.tensor(0).to(device))
    P = torch.randn((d, d), device=device)

    corr_u = torch.load("corr_u.pt", map_location=device)
    mu_u = torch.load("mu_u.pt", map_location=device)
    Q = corr_u - mu_u.unsqueeze(1) @ mu_u.unsqueeze(0)

    decoder.eval()
    for k in tqdm(range(n), "Filtering Samples"):
        # prepare context
        idx_context = idx[k : k - 1 + block_size].to(device)
        with torch.no_grad():
            x_context = decoder.token_embedding_table.to(device)(idx_context)
        x_k = torch.vstack((x_context, x_hat)).unsqueeze(0)
        y_k = Y[k, :].to(device)

        # prediction step
        x_pred, F = decoder.val_and_grad(
            x_k=x_k,
            encoder_output=encoder_output[: x_k.shape[0], : x_k.shape[1], :],
        )
        x_pred = x_pred + mu_u  # add the bias
        P_pred = F @ P @ F.T + Q

        # kalman gain
        K = torch.linalg.solve(P_pred + R, P_pred, left=False)

        # observation update step
        x_hat = x_pred + K @ (y_k - x_pred)
        P = (eye - K) @ P_pred @ (eye - K).T + K @ R @ K.T

        X_hat[k, :] = x_hat.detach().cpu().numpy()

    torch.save(X, "X_ekf.pt")
    torch.save(Y, "Y_ekf.pt")
    torch.save(X_hat, "X_hat_ekf.pt")
