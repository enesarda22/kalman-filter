import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

if __name__ == "__main__":
    n = 100000
    sig_u = np.sqrt(0.5)
    sig_w = 2

    freq_grid = np.logspace(-2, 2, 100)
    mse_values = np.empty(freq_grid.shape, dtype=float)
    for i, freq in tqdm(enumerate(freq_grid), "Frequencies"):
        normalization_c = np.sqrt(1 / (1 - np.exp(-2 * freq ** 2)))

        def state_func(x):
            return torch.sin(freq * x) * normalization_c

        x = np.random.randn(n)
        for k in range(1, n):
            x[k] = state_func(torch.tensor(x[k-1])).item() + sig_u * np.random.randn()
        y = x[1:] + sig_w * np.random.randn(n-1)

        # extended kalman filter
        filtered_y = np.empty(n-1, dtype=float)

        x_hat = torch.tensor(0.0, requires_grad=True)
        P = torch.tensor(1.0, requires_grad=True)
        for k in range(1, n):
            x_pred = state_func(x_hat)
            x_pred.backward()

            F_tilda = x_hat.grad
            P_pred = F_tilda * P * F_tilda + sig_u ** 2
            K = P_pred / (P_pred + sig_w ** 2)

            x_hat = x_pred + K * (y[k-1] - x_pred)
            x_hat = x_hat.detach().requires_grad_()
            P = (1 - K) * P_pred * (1 - K) + K * (sig_w ** 2) * K

            filtered_y[k-1] = x_hat.item()

        mse_values[i] = np.mean((x[1:] - filtered_y) ** 2)

    plt.loglog(freq_grid, mse_values)

    plt.xlabel(r'$\omega$')
    plt.ylabel("MSE")

    plt.grid(which="both", axis="y")
    plt.grid(which="major", axis="x")
    plt.show()
