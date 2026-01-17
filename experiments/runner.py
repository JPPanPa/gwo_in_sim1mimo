import numpy as np
from problem import SIM1MIMO_Problem
from utils.comm import (
    rayleigh_channel, noise_var_from_snr_db,
    unpack_w, normalize_w, sinr, mrc_w, ber_qpsk
)

def run_comparison(algorithms, cfg):

    results = {name: {} for name in algorithms.keys()}
    results["MRC"] = {}

    for snr_db in cfg.snr_db_list:
        sigma2 = noise_var_from_snr_db(snr_db)

        # init per SNR for ALL methods (meta + MRC)
        for name in results.keys():
            results[name][snr_db] = {"sinr_list": [], "ber_list": [], "time_list": []}

        # Monte Carlo frames (random channels)
        for _ in range(cfg.n_frames):
            h = rayleigh_channel(cfg.M)

            # Baseline: MRC (SIMO optimal linear combiner in AWGN)
            w_mrc = mrc_w(h)
            sinr_mrc = sinr(w_mrc, h, sigma2)
            ber_mrc = ber_qpsk(w_mrc, h, sigma2, cfg.n_syms_ber)

            results["MRC"][snr_db]["sinr_list"].append(sinr_mrc)
            results["MRC"][snr_db]["ber_list"].append(ber_mrc)
            results["MRC"][snr_db]["time_list"].append(0.0)
            # Metaheuristics: optimize w

            prob = SIM1MIMO_Problem(M=cfg.M, sigma2=sigma2)
            prob.set_channel(h, sigma2)

            for alg_key, (_display_name, AlgClass, kwargs) in algorithms.items():
                # safety: skip if Algo class missing
                if AlgClass is None:
                    continue

                alg = AlgClass(prob, n_wolves=cfg.n_wolves, max_iter=cfg.max_iter, **kwargs)
                best_sol, _best_fit = alg.optimize()

                w_best = normalize_w(unpack_w(best_sol))
                sinr_best = sinr(w_best, h, sigma2)
                ber_best = ber_qpsk(w_best, h, sigma2, cfg.n_syms_ber)

                results[alg_key][snr_db]["sinr_list"].append(sinr_best)
                results[alg_key][snr_db]["ber_list"].append(ber_best)
                results[alg_key][snr_db]["time_list"].append(alg.exec_time)

    # Aggregate summary
    summary = {name: {} for name in results.keys()}

    for name in results.keys():
        for snr_db in cfg.snr_db_list:
            s = np.array(results[name][snr_db]["sinr_list"], dtype=float)
            b = np.array(results[name][snr_db]["ber_list"], dtype=float)
            t = np.array(results[name][snr_db]["time_list"], dtype=float)

            summary[name][snr_db] = {
                "sinr_mean_db": float(10 * np.log10(np.mean(s))),
                "ber_mean": float(np.mean(b)),
                "time_mean": float(np.mean(t)),
            }

    return summary
