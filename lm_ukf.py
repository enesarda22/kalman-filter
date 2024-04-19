import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from models.semantic_decoder import SemanticDecoder
from models.semantic_encoder import SemanticEncoder
from models.semantic_transformer import SemanticTransformer

from utils.dataloader import Dataloader
from utils.general import get_device


def cholesky(A):
    """Find the nearest positive-definite matrix to input

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    B = (A + A.T) / 2

    _, s, V = torch.linalg.svd(B)
    H = V.T @ torch.diag(s) @ V

    A2 = (B + H) / 2
    A3 = (A2 + A2.T) / 2

    try:
        C = torch.linalg.cholesky(A3)
        return C
    except torch.linalg.LinAlgError:
        eps = torch.finfo(torch.float32).eps

        I = torch.eye(A.shape[0], device=device)
        k = 1
        while True:
            mineig = torch.min(torch.real(torch.linalg.eigvals(A3)))
            A3 += I * (-mineig * k**2 + eps)
            k += 1

            try:
                C = torch.linalg.cholesky(A3)
                return C
            except torch.linalg.LinAlgError:
                continue


def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = torch.linalg.cholesky(B)
        return True
    except torch.linalg.LinAlgError:
        return False


if __name__ == "__main__":
    device = get_device()
    torch.manual_seed(42)

    alpha = 1.0
    kappa = 384.0
    beta = 0.0

    # hyperparameters
    batch_size = 64
    block_size = 64
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

    idx = dataloader.get_batch(split="test", random=False)
    idx = torch.cat((idx[0, :], idx[1 : 30 * batch_size, -1]))

    # transmitted and received signals
    with torch.no_grad():
        X = transformer.semantic_decoder.token_embedding_table.to("cpu")(idx)
    Y = X[block_size - 1 :, :] + sig_w * torch.randn(X[block_size - 1 :, :].shape)

    n, d = Y.shape

    # initialize for UKF
    X_hat = np.empty((n, d), dtype=float)
    eye = torch.eye(d, device=device)

    x_hat = X[block_size - 1, :].to(device)
    P = torch.eye(d, device=device)

    corr_u = torch.load("corr_u_ukf.pt", map_location=device)
    mu_u = torch.load("mu_u_ukf.pt", map_location=device)
    Q = corr_u - mu_u.unsqueeze(1) @ mu_u.unsqueeze(0)

    L = 2 * d
    lambda_ = (alpha**2) * (L + kappa) - L
    w_m = (1 / (2 * (L + lambda_))) * torch.ones(2 * L + 1, device=device)
    w_m[0] = lambda_ / (L + lambda_)
    w_c = (1 / (2 * (L + lambda_))) * torch.ones(2 * L + 1, device=device)
    w_c[0] = lambda_ / (L + lambda_) + (1 - alpha**2 + beta)

    transformer.to(device)
    transformer.eval()
    for k in tqdm(range(n), "Filtering Samples"):
        # prepare context
        x_context = X[k : k - 1 + block_size].to(device)
        # idx_context = idx[k : k - 1 + block_size].to(device)
        # with torch.no_grad():
        #     x_context = transformer.semantic_decoder.token_embedding_table.to(device)(
        #         idx_context
        #     )
        y_k = Y[k, :].to(device)

        x_hat_a = torch.cat([x_hat, mu_u])
        P_a = torch.block_diag(P, Q)

        delta = cholesky((L + lambda_) * P_a)
        sigma_mat = torch.hstack(
            [x_hat_a[:, None], x_hat_a[:, None] + delta, x_hat_a[:, None] - delta]
        ).T

        # prediction step
        x_k = torch.cat(
            [x_context[None, :, :].expand(2 * L + 1, -1, -1), sigma_mat[:, None, :d]],
            dim=1,
        )
        u_k = sigma_mat[:, None, d:]

        dataset = TensorDataset(x_k, u_k)
        dl = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        sigma_pred = []
        for x_b, u_b in dl:
            with torch.no_grad():
                sigma_pred_b = transformer.semantic_decoder.val(x_k=x_b, u_k=u_b)
            sigma_pred.append(sigma_pred_b.detach().cpu())
        sigma_pred = torch.cat(sigma_pred).to(device)

        x_pred = w_m @ sigma_pred

        dev = sigma_pred[:, :, None] - x_pred[:, None]
        P_pred = (dev @ dev.transpose(1, 2)).transpose(0, 2) @ w_c

        # kalman gain
        K = torch.linalg.solve(P_pred + R, P_pred, left=False)

        # observation update step
        x_hat = x_pred + K @ (y_k - x_pred)
        P = (eye - K) @ P_pred @ (eye - K).T + K @ R @ K.T
        # P = P_pred - K @ (P_pred + R) @ K.T

        x_hat = transformer.semantic_decoder.get_close_embeddings(x_hat[None, :])[0, :]
        X_hat[k, :] = x_hat.detach().cpu().numpy()

    torch.save(X, "X_ukf.pt")
    torch.save(Y, "Y_ukf.pt")
    torch.save(X_hat, "X_hat_ukf.pt")
