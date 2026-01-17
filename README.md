# Grey Wolf Optimization for SIM-1-MIMO

**Đồ án Nhập môn Kỹ thuật Truyền thông - Nhóm 121**

Ứng dụng GWO và các biến thể để tối ưu vector kết hợp thu trong hệ thống SIMO (1 Tx, M=8 Rx).

---

## Bài toán

**Hệ thống**: 1 anten phát, M=8 anten thu, kênh Rayleigh  
**Mục tiêu**: Tối ưu vector kết hợp **w ∈ ℂ^M** để maximize SINR

```
SINR(w) = |w^H·h|² / (σ²·||w||²)

```

**Thuật toán**:
- **GWO**: Grey Wolf Optimizer (gốc)
- **OBL-GWO**: Opposition-Based Learning
- **LF-GWO**: Levy Flight
- **Chaotic-GWO**: Chaotic Maps
- **MRC**: Maximum Ratio Combining (baseline)

---

##  Cấu trúc

```
GWO-SIMO/
├── algorithms/         # GWO, OBL-GWO, LF-GWO, Chaotic-GWO
├── problem/           # SIM1MIMO_Problem
├── utils/             # comm.py (channel, SINR, BER)
├── experiments/       # runner.py (Monte Carlo)
├── plots/             # plotter.py
├── config.py          # Tham số
└── main.py            # Entry point
```

---

##  Cài đặt

```bash
pip install numpy matplotlib
```

---

## Sử dụng

### Chạy nhanh

```bash
python main.py
```

**Output**: Bảng SINR/BER + 2 biểu đồ PNG

### Tùy chỉnh tham số

File `config.py`:

```python
M: int = 8                  # Số anten thu
snr_db_list = (0,5,10,15,20)  # SNR points
n_wolves: int = 25          # Kích thước quần thể
max_iter: int = 120         # Số vòng lặp
```

### Code mẫu

```python
from problem import SIM1MIMO_Problem
from algorithms import GWO
from utils.comm import rayleigh_channel, mrc_w, sinr

# Tạo bài toán
h = rayleigh_channel(M=8)
problem = SIM1MIMO_Problem(M=8, h=h, sigma2=1e-3)

# Chạy GWO
gwo = GWO(problem, n_wolves=30, max_iter=200)
w_opt, _ = gwo.optimize()

# So sánh với MRC
w_mrc = mrc_w(h)
print(f"GWO SINR: {sinr(w_opt, h, 1e-3):.2f}")
print(f"MRC SINR: {sinr(w_mrc, h, 1e-3):.2f}")
```

---

## Kết quả

**SINR trung bình (dB)**:

| SNR | GWO   | OBL-GWO | LF-GWO | Chaotic | MRC   |
|-----|-------|---------|--------|---------|-------|
| 0   | 8.79  | 8.70    | 6.47   | 8.62    | 8.81  |
| 5   | 14.51 | 14.42   | 12.35  | 14.37   | 14.52 |
| 10  | 18.95 | 18.86   | 17.02  | 18.79   | 18.97 |
| 15  | 23.52 | 23.43   | 20.96  | 23.35   | 23.53 |
| 20  | 29.02 | 28.93   | 26.88  | 28.86   | 29.04 |

**Kết luận**:
-  GWO tiệm cận MRC (gap < 0.2 dB)
-  OBL, Chaotic tương đương GWO
- LF-GWO: exploration mạnh nhưng giảm convergence

---

## 🔬 Các hàm chính

**`utils/comm.py`**:
```python
rayleigh_channel(M)          # Tạo kênh Rayleigh
sinr(w, h, sigma2)           # Tính SINR
ber_qpsk(w, h, sigma2, N)    # Đo BER QPSK
mrc_w(h)                     # MRC combining: w = h/||h||
```

**`problem/sim1mimo_problem.py`**:
```python
class SIM1MIMO_Problem:
    def fitness(self, x):    # x ∈ ℝ^(2M) → -SINR(w)
    def get_bounds(self):    # [-1,1]^(2M)
```

---

##  Tham khảo

1. Mirjalili et al., "Grey Wolf Optimizer", *Advances in Engineering Software*, 2014
2. Báo cáo đồ án: `GWO_Report.pdf` (24 trang)
