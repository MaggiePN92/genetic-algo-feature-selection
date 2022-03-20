import random
from src.individual import Individual


class Generation:
    def __init__(self, pop_size, n_genes, score_func, crossover_vecs = None):
        self.pop_size = pop_size
        self.n_genes = n_genes
        self.score_func = score_func
        self.pop = []

        if not crossover_vecs:
            self._init_pop()
        
        else:
            self.pop = crossover_vecs


    def _init_pop(self):
        for i in range(self.pop_size):
            vec = [random.randint(0, 1) for _ in range(self.n_genes)]
            fitness = self.score_func(vec)
            
            indi = Individual(i, vec, fitness)
            self.pop.append(indi)

    def _sort_generation(self):
        self.pop.sort(key = lambda x: x.fitness)

    def mutate(self):
        raise NotImplemented



    # def _init_pop(self):
    #     """
    #     Lager en ny populasjon med individer ved en ny generasjon. Tar hensyn til crossover individer.

    #     :return:
    #     """
    #     if len(self.crossover_vecs) > 0:
    #         for idx, ind in enumerate(self.crossover_vecs):
    #             self.pop.append(
    #                 {
    #                     "idx" : idx,
    #                     "vec": ind["vec"],
    #                     "fitness": ind["fitness"],
    #                  }
    #             )

    #     for i in range(len(self.crossover_vecs), self.pop_size):
    #         vec = [random.randint(0, 1) for _ in range(self.n_genes)]

    #         self.pop.append({
    #             "idx": i,
    #             "vec" : vec,
    #             "fitness" : self.score_func(vec)
    #         })
