import random
import numpy as np


class Generation:
    def __init__(self, pop_size, vec_len, score_func, crossover_vecs = []):
        self.pop_size = pop_size
        self.vec_len = vec_len
        self.score_func = score_func
        self.pop = []
        self.crossover_vecs = crossover_vecs
        self._init_pop()


    def _init_pop(self):
        """
        Lager en ny populasjon med individer ved en ny generasjon. Tar hensyn til crossover individer.

        :return:
        """
        if len(self.crossover_vecs) > 0:
            for idx, ind in enumerate(self.crossover_vecs):
                self.pop.append(
                    {
                        "idx" : idx,
                        "vec": ind["vec"],
                        "fitness": ind["fitness"],
                     }
                )

        for i in range(len(self.crossover_vecs), self.pop_size):
            vec = [random.randint(0, 1) for _ in range(self.vec_len)]

            self.pop.append({
                "idx": i,
                "vec" : vec,
                "fitness" : self.score_func(vec)
            })

    def mutate(self):
        raise NotImplemented
