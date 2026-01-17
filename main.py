from config import Config
from algorithms import GWO, OBL_GWO, LF_GWO, Chaotic_GWO
from experiments.runner import run_comparison
from plots.plotter import plot_results

def build_algorithms(cfg: Config):
    return {
        "GWO": ("GWO", GWO, {}),
        "OBL-GWO": ("OBL-GWO", OBL_GWO, {"jump_rate": cfg.obl_jump_rate}),
        "LF-GWO": ("LF-GWO", LF_GWO, {"levy_prob": cfg.lf_levy_prob}),
        "Chaotic-GWO": ("Chaotic-GWO", Chaotic_GWO, {"chaos_type": cfg.chaos_type}),
    }

if __name__ == "__main__":
    cfg = Config()
    algos = build_algorithms(cfg)

    print("=" * 70)
    print("SIM-1-MIMO (SIMO) RECEIVE COMBINING: GWO & VARIANTS")
    print("=" * 70)

    summary = run_comparison(algos, cfg)

    print("\nSNR(dB) | " + " | ".join([f"{name:12s} SINR(dB) / BER" for name in summary.keys()]))
    print("-" * 110)
    for snr in cfg.snr_db_list:
        row = [f"{snr:>6}"]
        for name in summary.keys():
            sinr_db = summary[name][snr]["sinr_mean_db"]
            ber = summary[name][snr]["ber_mean"]
            row.append(f"{sinr_db:>7.2f} / {ber:.3e}")
        print(" | ".join(row))

    plot_results(summary, cfg)
