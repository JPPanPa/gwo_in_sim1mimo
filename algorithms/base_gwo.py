import numpy as np
from utils.timing import timer

class BaseGWO:
    def __init__(self, obj_func, n_wolves=30, max_iter=500):
        self.obj_func = obj_func
        self.n_wolves = n_wolves
        self.max_iter = max_iter
        self.dim = obj_func.dim
        self.lb, self.ub = obj_func.get_bounds()

        self.convergence = []
        self.best_solution = None
        self.best_fitness = float("inf")
        self.exec_time = 0.0

    def initialize_population(self):
        return np.random.uniform(self.lb, self.ub, (self.n_wolves, self.dim))

    def clip(self, x):
        return np.clip(x, self.lb, self.ub)

    def eval_fitness(self, wolves):
        return np.array([self.obj_func.fitness(w) for w in wolves])

    def select_leaders(self, wolves, fitness):
        idx = np.argsort(fitness)
        alpha_pos = wolves[idx[0]].copy()
        alpha_score = fitness[idx[0]]
        beta_pos = wolves[idx[1]].copy()
        delta_pos = wolves[idx[2]].copy()
        return alpha_pos, alpha_score, beta_pos, delta_pos, idx

    def update_positions(self, wolves, alpha_pos, beta_pos, delta_pos, a):
        # GWO core update
        for i in range(self.n_wolves):
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

    def post_iteration_hook(self, wolves, fitness, iter_idx):
        # for variants to override (OBL jump, chaotic params, ...)
        return wolves, fitness

    def optimize(self):
        with timer() as elapsed:
            wolves = self.initialize_population()
            fitness = self.eval_fitness(wolves)

            alpha_pos, alpha_score, beta_pos, delta_pos, _ = self.select_leaders(wolves, fitness)

            for it in range(self.max_iter):
                a = 2 - it * (2.0 / self.max_iter)

                wolves = self.update_positions(wolves, alpha_pos, beta_pos, delta_pos, a)
                fitness = self.eval_fitness(wolves)

                wolves, fitness = self.post_iteration_hook(wolves, fitness, it)

                alpha_pos2, alpha_score2, beta_pos2, delta_pos2, idx = self.select_leaders(wolves, fitness)
                if alpha_score2 < alpha_score:
                    alpha_pos, alpha_score = alpha_pos2, alpha_score2
                beta_pos, delta_pos = beta_pos2, delta_pos2

                self.convergence.append(alpha_score)

            self.best_solution = alpha_pos
            self.best_fitness = alpha_score
            self.exec_time = elapsed()

        return self.best_solution, self.best_fitness
