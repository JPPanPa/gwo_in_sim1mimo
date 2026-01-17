import numpy as np

def rayleigh_channel(M: int):
    return (np.random.randn(M) + 1j * np.random.randn(M)) / np.sqrt(2)

def noise_var_from_snr_db(snr_db: float):
    snr_lin = 10 ** (snr_db / 10.0)
    return 1.0 / snr_lin  # assume E[|x|^2] = 1

def pack_w(w: np.ndarray):
    return np.concatenate([np.real(w), np.imag(w)])

def unpack_w(x: np.ndarray):
    M2 = x.shape[0]
    M = M2 // 2
    w = x[:M] + 1j * x[M:]
    return w

def normalize_w(w: np.ndarray):
    n = np.linalg.norm(w)
    if n == 0:
        return w
    return w / n

def sinr(w: np.ndarray, h: np.ndarray, sigma2: float):
    # SINR = |w^H h|^2 / (sigma2 * ||w||^2)
    num = np.abs(np.vdot(w, h)) ** 2
    den = sigma2 * (np.linalg.norm(w) ** 2)
    return float(num / den)

def mrc_w(h: np.ndarray):
    return normalize_w(h.copy())

def qpsk_symbols(n: int):
    # (±1 ± j)/sqrt(2)
    b = np.random.randint(0, 2, size=(n, 2))
    re = 2*b[:,0] - 1
    im = 2*b[:,1] - 1
    x = (re + 1j*im) / np.sqrt(2)
    return x, b

def qpsk_detect(z: np.ndarray):
    re = (np.real(z) >= 0).astype(int)
    im = (np.imag(z) >= 0).astype(int)
    return np.stack([re, im], axis=1)

def ber_qpsk(w: np.ndarray, h: np.ndarray, sigma2: float, n_syms: int):
    w = normalize_w(w)
    x, b = qpsk_symbols(n_syms)
    M = h.shape[0]
    noise = (np.random.randn(n_syms, M) + 1j*np.random.randn(n_syms, M)) * np.sqrt(sigma2/2)
    y = x[:, None] * h[None, :] + noise
    z = y @ np.conjugate(w)  # z = w^H y
    b_hat = qpsk_detect(z)
    err = np.sum(b_hat != b)
    return err / b.size
