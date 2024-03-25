import numpy as np
from scipy.signal import lfilter
import matplotlib.pyplot as plt
from tqdm import tqdm

if __name__ == "__main__":
    sig_u = 1
    sig_w = 5

    b = np.array([0.25, 0.25, 0.25, 0.25])
    a = np.array([1, -0.6, -0.4])

    q = len(b) - 1
    p = len(a) - 1

    u = np.random.randn(100000) * sig_u
    x = lfilter(b, a, u)
    x_observed = x + np.random.randn(len(x)) * sig_w

    plt.plot(x, label="ARMA(2, 3) process")
    plt.legend()
    plt.show()

    F = np.block([[-a[1:]], [np.eye(p-1), np.zeros((p-1))]])
    G = np.block([[b], [np.zeros((p-1, q+1))]])
    Q = (sig_u ** 2) * np.eye(q+1)

    C = np.eye(p)
    R = np.eye(p) * sig_w

    x_hat = np.zeros(p)
    P = np.eye(p)

    x_new = np.empty(len(u))
    for i in tqdm(range(len(u))):
        # predict
        x_pred = F @ x_hat
        r_pred = C @ x_pred

        P_pred = F @ P @ F.T + G @ Q @ G.T
        P_tilda = C @ P_pred @ C.T + R

        # kalman gain
        temp = np.linalg.lstsq(P_tilda.T, C)[0]
        K = P_pred @ temp.T

        # estimation
        x_hat = F @ x_hat + K @ (x_observed[i] - C @ F @ x_hat)
        P = (np.eye(p) - K @ C) @ P_pred

        x_new[i] = x_hat[-1]

    print(f"Observed MSE: {np.mean((x_observed - x) ** 2)}")
    print(f"Filtered MSE: {np.mean((x_new - x) ** 2)}")

    plt.plot(x_observed[10000:10100], label="Observed")
    plt.plot(x_new[10000:10100], label="Filtered")
    plt.legend()
    plt.show()




