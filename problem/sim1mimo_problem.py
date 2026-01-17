import numpy as np
from utils.comm import unpack_w, normalize_w, sinr

class SIM1MIMO_Problem:
    """
    SIM-1-MIMO (SIMO): 1 Tx, M Rx
    Optimize receive combining vector w (complex Mx1)
    Decision variable x in R^(2M): [Re(w), Im(w)]
    Objective: maximize SINR => minimize -SINR
    """

    def __init__(self, M=8, h=None, sigma2=1e-3):
        self.M = M
        self.dim = 2 * M
        self.sigma2 = sigma2
        self.h = h  # set per frame

    def set_channel(self, h, sigma2):
        self.h = h
        self.sigma2 = sigma2

    def fitness(self, x):
        w = normalize_w(unpack_w(x))
        # minimize negative SINR
        return -sinr(w, self.h, self.sigma2)

    def get_bounds(self):
        lb = -1.0 * np.ones(self.dim)
        ub =  1.0 * np.ones(self.dim)
        return lb, ub
