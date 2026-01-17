import numpy as np
from .base_gwo import BaseGWO

class Chaotic_GWO(BaseGWO):
    def __init__(self, obj_func, n_wolves=30, max_iter=500, chaos_type="logistic"):
        super().__init__(obj_func, n_wolves, max_iter)
        self.chaos_type = chaos_type
        self.c = np.random.random()  # chaos state

    def chaotic_map(self, x):
        if self.chaos_type == "logistic":
            return 4 * x * (1 - x)
        if self.chaos_type == "sine":
            return np.sin(np.pi * x)
        if self.chaos_type == "tent":
            return x/0.7 if x < 0.7 else (1-x)/0.3
        return x

    def initialize_population(self):
        pop = np.zeros((self.n_wolves, self.dim))
        c = np.random.random(self.dim)
        for i in range(self.n_wolves):
            c = self.chaotic_map(c)
            pop[i] = self.lb + c * (self.ub - self.lb)
        return pop

    def optimize(self):
        # override only to change 'a' by chaos; keep code minimal
        import numpy as np
        from utils.timing import timer

        with timer() as elapsed:
            wolves = self.initialize_population()
            fitness = self.eval_fitness(wolves)
            alpha_pos, alpha_score, beta_pos, delta_pos, _ = self.select_leaders(wolves, fitness)

            for it in range(self.max_iter):
                self.c = self.chaotic_map(self.c)
                a = 2 * self.c  # chaotic a

                wolves = self.update_positions(wolves, alpha_pos, beta_pos, delta_pos, a)
                fitness = self.eval_fitness(wolves)

                alpha_pos2, alpha_score2, beta_pos2, delta_pos2, _ = self.select_leaders(wolves, fitness)
                if alpha_score2 < alpha_score:
                    alpha_pos, alpha_score = alpha_pos2, alpha_score2
                beta_pos, delta_pos = beta_pos2, delta_pos2

                self.convergence.append(alpha_score)

            self.best_solution = alpha_pos
            self.best_fitness = alpha_score
            self.exec_time = elapsed()

        return self.best_solution, self.best_fitness
