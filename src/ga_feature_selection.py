from numpy import dot
from numpy.linalg import norm


def scoring_func(vec):
    soln = [0, 1, 0, 1, 0]
    cos_sim = dot(soln, vec) / (norm(soln) * norm(vec))
    return cos_sim


class GA:
    def __init__(self, n_ind, repl_frac, mutation_rate, scoring_func):
        self.n_ind = n_ind
        self.repl_frac = repl_frac
        self.mutation_rate = mutation_rate
        self.scoring_func = scoring_func

