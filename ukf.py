import numpy as np

from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from tqdm import tqdm

if __name__ == "__main__":
    alpha = 0.5
    kappa = 2.0
    beta = 2.0

    n = 100000
    sig_u = np.sqrt(0.5)
    sig_w = 2

    freq_grid = np.logspace(-2, 2, 100)
    mse_values = np.empty(freq_grid.shape, dtype=float)
    mse_values_ekf = np.empty(freq_grid.shape, dtype=float)
    mse_values_obs = np.empty(freq_grid.shape, dtype=float)
    for i, freq in tqdm(enumerate(freq_grid), "Frequencies"):
        normalization_c = np.sqrt(1 / (1 - np.exp(-2 * freq**2)))

        def state_func(x):
            return np.sin(freq * x) * normalization_c

        x = np.random.randn(n)
        for k in range(1, n):
            x[k] = state_func(x[k - 1]) + sig_u * np.random.randn()
        y = x[1:] + sig_w * np.random.randn(n - 1)

        # unscented kalman filter
        filtered_y = np.empty(n - 1, dtype=float)
        obs_x_hat = y * (1 / (1 + sig_w**2))
        filtered_y_ekf = np.empty(n - 1, dtype=float)

        x_hat = 0.0
        P = 1.0

        x_hat_ekf = 0.0
        P_ekf = 1.0

        L = 2
        lambda_ = (alpha**2) * (L + kappa) - L

        w_m = (1 / (2 * (L + lambda_))) * np.ones(2 * L + 1)
        w_m[0] = lambda_ / (L + lambda_)

        w_c = (1 / (2 * (L + lambda_))) * np.ones(2 * L + 1)
        w_c[0] = lambda_ / (L + lambda_) + (1 - alpha**2 + beta)

        for k in range(1, n):
            x_hat_a = np.array([x_hat, 0.0])
            P_a = np.diag([P, sig_u**2])

            delta = np.linalg.cholesky((L + lambda_) * P_a)
            sigma_mat = np.hstack(
                [x_hat_a[:, None], x_hat_a[:, None] + delta, x_hat_a[:, None] - delta]
            ).T

            # prediction
            sigma_pred = state_func(sigma_mat[:, 0]) + sigma_mat[:, 1]
            x_pred = w_m @ sigma_pred
            P_pred = w_c @ ((sigma_pred - x_pred) * (sigma_pred - x_pred))
            K = P_pred / (P_pred + sig_w**2)

            x_pred_ekf = state_func(x_hat_ekf)
            F_tilda = normalization_c * freq * np.cos(freq * x_hat)
            P_pred_ekf = F_tilda * P_ekf * F_tilda + sig_u**2
            K_ekf = P_pred_ekf / (P_pred_ekf + sig_w**2)

            # correction
            x_hat = x_pred + K * (y[k - 1] - x_pred)
            P = (1 - K) * P_pred * (1 - K) + K * (sig_w**2) * K

            x_hat_ekf = x_pred_ekf + K_ekf * (y[k - 1] - x_pred_ekf)
            P_ekf = (1 - K_ekf) * P_pred_ekf * (1 - K_ekf) + K_ekf * (sig_w**2) * K_ekf

            filtered_y[k - 1] = x_hat
            filtered_y_ekf[k - 1] = x_hat_ekf

        mse_values[i] = np.mean((x[1:] - filtered_y) ** 2)
        mse_values_obs[i] = np.mean((x[1:] - obs_x_hat) ** 2)
        mse_values_ekf[i] = np.mean((x[1:] - filtered_y_ekf) ** 2)

    fig, ax = plt.subplots()

    ax.loglog(freq_grid, mse_values_ekf, label="EKF")
    ax.plot(freq_grid, mse_values_obs, label="Linear Estimator")
    ax.loglog(freq_grid, mse_values, "r", label="UKF")

    ax.set_title("MSE at Different Frequencies")
    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel("MSE")

    ax.grid(which="both", axis="y")
    ax.grid(which="major", axis="x")

    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.yaxis.set_minor_formatter(FormatStrFormatter("%.1f"))

    ax.legend()
    plt.show()
