from dataclasses import dataclass

@dataclass
class Config:
    # SIM-1-MIMO (SIMO): 1 Tx, M Rx
    M: int = 8  # số anten thu

    # Monte Carlo
    snr_db_list: tuple = (0, 5, 10, 15, 20)
    n_frames: int = 30           # số kênh ngẫu nhiên mỗi SNR
    n_syms_ber: int = 4000       # số symbol để ước lượng BER

    # Algorithms
    n_runs: int = 3              # số lần lặp thí nghiệm (giảm để chạy nhanh)
    n_wolves: int = 25
    max_iter: int = 120

    # Variant params
    obl_jump_rate: float = 0.3
    lf_levy_prob: float = 0.3
    chaos_type: str = "logistic"

    # Output
    fig_path_prefix: str = "sim1mimo_results"
