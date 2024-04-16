import numpy as np
from scipy.stats import norm

import seaborn as sns
from matplotlib import pyplot as plt

if __name__ == "__main__":
    x = np.random.randn(1000000)

    freq = 2 * np.pi
    normalization_c = np.sqrt(1 / (1 - np.exp(-2 * freq**2)))
    y = np.sin(freq * x) * normalization_c

    sig_y = np.std(y)
    mu_y = np.mean(y)
    grid = np.linspace(-1.5, 1.5, 10000)
    gaussian_y = norm.pdf(grid, loc=mu_y, scale=sig_y)

    # plot distribution of y
    _, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(y, stat="density", ax=ax, label="$p(y)$")
    sns.lineplot(x=grid, y=gaussian_y, label="Gaussian of $p(y)$")

    plt.title("Distribution of $c \\times sin(2 \pi X)$")
    plt.xlabel("Value")
    plt.ylabel("Density")

    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.set_xlim([-1.05, 1.05])

    plt.legend()
    plt.show()

    # EKF
    mu_y_ekf = 0
    sig_y_ekf = np.sqrt(normalization_c * freq * np.cos(freq * mu_y_ekf))
    gaussian_y_ekf = norm.pdf(grid, loc=mu_y_ekf, scale=sig_y_ekf)

    _, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(y, stat="density", ax=ax, label="$p(y)$")
    sns.lineplot(x=grid, y=gaussian_y, label="Gaussian of $p(y)$")
    sns.lineplot(x=grid, y=gaussian_y_ekf, label="EKF Gaussian")

    plt.title("Distribution of $c \\times sin(2 \pi X)$")
    plt.xlabel("Value")
    plt.ylabel("Density")

    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.set_xlim([-1.05, 1.05])

    plt.legend()
    plt.show()

    # UKF
    alpha = 0.5
    kappa = 2.0
    beta = 2.0
    L = 1
    lambda_ = (alpha**2) * (L + kappa) - L
    w_m = (1 / (2 * (L + lambda_))) * np.ones(2 * L + 1)
    w_m[0] = lambda_ / (L + lambda_)
    w_c = (1 / (2 * (L + lambda_))) * np.ones(2 * L + 1)
    w_c[0] = lambda_ / (L + lambda_) + (1 - alpha**2 + beta)

    delta = np.sqrt(L + lambda_)
    sigma_x = np.array([0, delta, -delta])
    sigma_y = np.sin(freq * sigma_x) * normalization_c

    mu_y_ukf = w_m @ sigma_y
    sig_y_ukf = np.sqrt(w_c @ ((sigma_y - mu_y_ukf) ** 2))
    gaussian_y_ukf = norm.pdf(grid, loc=mu_y_ukf, scale=sig_y_ukf)

    _, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(y, stat="density", ax=ax, label="$p(y)$")
    sns.lineplot(x=grid, y=gaussian_y, label="Gaussian of $p(y)$")
    sns.lineplot(x=grid, y=gaussian_y_ekf, label="EKF Gaussian")
    sns.lineplot(x=grid, y=gaussian_y_ukf, color="r", label="UKF Gaussian")
    sns.scatterplot(
        x=sigma_x, y=[0, 0, 0], marker="x", s=90, color="k", label="Sigma Points"
    )

    plt.title("Distribution of $c \\times sin(2 \pi X)$")
    plt.xlabel("Value")
    plt.ylabel("Density")

    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.set_xlim([-1.05, 1.05])

    plt.legend()
    plt.show()

    # plot distribution of x
    _, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(x, stat="density", ax=ax)

    plt.title("Distribution of $X$")
    plt.xlabel("Value")
    plt.ylabel("Density")

    ax.grid(axis="y")
    ax.set_axisbelow(True)
    plt.show()
