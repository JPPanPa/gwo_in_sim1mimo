import numpy as np
from .base_gwo import BaseGWO

class OBL_GWO(BaseGWO):
    def __init__(self, obj_func, n_wolves=30, max_iter=500, jump_rate=0.3):
        super().__init__(obj_func, n_wolves, max_iter)
        self.jump_rate = jump_rate

    def opposition_population(self, pop):
        return self.lb + self.ub - pop

    def initialize_population(self):
        pop = np.random.uniform(self.lb, self.ub, (self.n_wolves, self.dim))
        opp = self.opposition_population(pop)
        combined = np.vstack([pop, opp])
        fit = np.array([self.obj_func.fitness(x) for x in combined])
        idx = np.argsort(fit)
        return combined[idx[:self.n_wolves]]

    def post_iteration_hook(self, wolves, fitness, iter_idx):
        if np.random.random() < self.jump_rate:
            opp = self.opposition_population(wolves)
            combined = np.vstack([wolves, opp])
            fit = np.array([self.obj_func.fitness(x) for x in combined])
            idx = np.argsort(fit)
            wolves = combined[idx[:self.n_wolves]]
            fitness = fit[idx[:self.n_wolves]]
        return wolves, fitness
