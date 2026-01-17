import numpy as np
import matplotlib.pyplot as plt

def plot_results(summary, cfg):
    snrs = list(cfg.snr_db_list)
    names = list(summary.keys())

    # 1) SINR(dB) vs SNR(dB)
    plt.figure(figsize=(8, 5))
    for name in names:
        y = [summary[name][s]["sinr_mean_db"] for s in snrs]
        plt.plot(snrs, y, marker="o", label=name)
    plt.xlabel("SNR (dB)")
    plt.ylabel("Average SINR (dB)")
    plt.title("SIM-1-MIMO: SINR vs SNR")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{cfg.fig_path_prefix}_sinr.png", dpi=300, bbox_inches="tight")

    # 2) BER vs SNR (log scale)
    plt.figure(figsize=(8, 5))
    for name in names:
        y = [summary[name][s]["ber_mean"] for s in snrs]
        plt.semilogy(snrs, y, marker="o", label=name)
    plt.xlabel("SNR (dB)")
    plt.ylabel("BER (QPSK)")
    plt.title("SIM-1-MIMO: BER vs SNR")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{cfg.fig_path_prefix}_ber.png", dpi=300, bbox_inches="tight")

    print(f"Saved: {cfg.fig_path_prefix}_sinr.png")
    print(f"Saved: {cfg.fig_path_prefix}_ber.png")
    plt.show()
