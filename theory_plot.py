import numpy as np
import matplotlib.pyplot as plt

def construct_oneNN(path="theory_ellipsoidal.npz"):

    # Load results
    results = np.load(path)

    # Unpack values
    R, S, bterm, RSD = results['R'], results['S'], results['bterms'], results['RSD']

    # construct 1NN CDF
    vol = 4 * np.pi * R**3 / 3
    def oneNN(n, b, s):
        cdf = 1 - np.exp(-n*vol + n*n/2 * b*b * bterm[s])
        return cdf

    return oneNN, R, S

def plot_oneNN(n, b, path="theory_ellipsoidal.npz"):

    # Grab oneNN function
    oneNN, R, S = construct_oneNN(path)

    plt.figure(figsize=(12,12))

    plt.subplot(2,1,1)
    plt.loglog(R, np.minimum(oneNN(n, b, 0), 1-oneNN(n, b, 0)))
    plt.loglog(R, np.minimum(oneNN(n, b, 1), 1-oneNN(n, b, 1)))
    plt.loglog(R, np.minimum(oneNN(n, b, 2), 1-oneNN(n, b, 2)))
    plt.ylim(1e-3)
    plt.xlabel(r'Distance ($h^{-1}$ Mpc)')
    plt.ylabel(r'Peaked CDF')
    plt.grid(alpha=0.3)

    plt.subplot(2,1,2)
    plt.semilogx(R, oneNN(n, b, 1)-oneNN(n, b, 0), label=f'{S[1]}-{S[0]}')
    plt.semilogx(R, oneNN(n, b, 2)-oneNN(n, b, 0), label=f'{S[2]}-{S[0]}')
    plt.xlabel(r'Distance ($h^{-1}$ Mpc)')
    plt.ylabel(r'Peaked CDF')
    plt.legend()
    plt.grid(alpha=0.3)


    plt.savefig(f"{path[:-4]}{n}.png")


if __name__ == "__main__":
    plot_oneNN(1e-5,1)
    plot_oneNN(1e-4,1)


