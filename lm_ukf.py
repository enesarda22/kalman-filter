import numpy as np
import torch
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
    Y = X[block_size:, :] + sig_w * torch.randn(X[block_size:, :].shape)

    n, d = Y.shape

    # initialize for UKF
    X_hat = np.empty((n, d), dtype=float)

    Rx = X[:, :, None] @ X[:, None, :]
    Rx = torch.mean(Rx, dim=0)
    mux = torch.mean(X, dim=0)
    P = Rx - (mux[:, None] @ mux[None, :])
    P = P.to(device)

    transformer.to(device)
    transformer.eval()
    for k in tqdm(range(n), "Filtering Samples"):
        # prepare context
        idx_k = idx[k : k + block_size + 1].to(device)
        y_k = Y[k, :].to(device)

        # prediction step
        with torch.no_grad():
            x_pred = transformer.val(idx=idx_k[None, :])[0, :]

        # kalman gain
        K = torch.linalg.solve(P + R, P, left=False)

        # observation update step
        x_hat = x_pred + K @ (y_k - x_pred)

        # x_hat = transformer.semantic_decoder.get_close_embeddings(x_hat[None, :])[0, :]
        X_hat[k, :] = x_hat.detach().cpu().numpy()

    torch.save(X, "X_ukf.pt")
    torch.save(Y, "Y_ukf.pt")
    torch.save(X_hat, "X_hat_ukf.pt")
