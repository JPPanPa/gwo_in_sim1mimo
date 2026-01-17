import numpy as np
import math
from .base_gwo import BaseGWO

class LF_GWO(BaseGWO):
    def __init__(self, obj_func, n_wolves=30, max_iter=500, levy_prob=0.3):
        super().__init__(obj_func, n_wolves, max_iter)
        self.levy_prob = levy_prob

    def levy_flight(self, beta=1.5):
        sigma = (math.gamma(1+beta) * np.sin(np.pi*beta/2) /
                 (math.gamma((1+beta)/2) * beta * 2**((beta-1)/2)))**(1/beta)
        u = np.random.randn(self.dim) * sigma
        v = np.random.randn(self.dim)
        step = u / (np.abs(v) ** (1/beta))
        return 0.01 * step

    def update_positions(self, wolves, alpha_pos, beta_pos, delta_pos, a):
     for i in range(self.n_wolves):
        if np.random.random() < self.levy_prob:
            wolves[i] = wolves[i] + self.levy_flight() * (alpha_pos - wolves[i])
            wolves[i] = self.clip(wolves[i])
        else:
            # GWO update cho đúng 1 sói i (copy từ BaseGWO nhưng chỉ cho i này)
            for j in range(self.dim):
                r1, r2 = np.random.random(), np.random.random()
                A1, C1 = 2 * a * r1 - a, 2 * r2
                D_alpha = abs(C1 * alpha_pos[j] - wolves[i, j])
                X1 = alpha_pos[j] - A1 * D_alpha

                r1, r2 = np.random.random(), np.random.random()
                A2, C2 = 2 * a * r1 - a, 2 * r2
                D_beta = abs(C2 * beta_pos[j] - wolves[i, j])
                X2 = beta_pos[j] - A2 * D_beta

                r1, r2 = np.random.random(), np.random.random()
                A3, C3 = 2 * a * r1 - a, 2 * r2
                D_delta = abs(C3 * delta_pos[j] - wolves[i, j])
                X3 = delta_pos[j] - A3 * D_delta

                wolves[i, j] = (X1 + X2 + X3) / 3.0

            wolves[i] = self.clip(wolves[i])

        return wolves

