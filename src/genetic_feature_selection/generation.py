from random import randint
from genetic_feature_selection.individual import Individual


class Generation:
    def __init__(self, pop_size, n_genes, score_func,
     crossover_vecs = [], mutation_prob = 0.1):
        self.pop_size = pop_size
        self.n_genes = n_genes
        self.score_func = score_func
        self.pop = []
        self.mutation_prob = mutation_prob
        self.crossover_vecs = crossover_vecs
        self._init_pop()
    
    def _init_pop(self):
        if len(self.crossover_vecs) > 0:
            for idx, vec in enumerate(self.crossover_vecs):
                fitness = self.score_func(vec)
                indi = Individual(idx, vec, fitness)
                self.pop.append(indi)

        for i in range(len(self.crossover_vecs), self.pop_size):
            vec = [randint(0, 1) for _ in range(self.n_genes)]
            fitness = self.score_func(vec)
            
            indi = Individual(i, vec, fitness)
            self.pop.append(indi)

    def sort_generation(self):
        self.pop.sort(key = lambda x: x.fitness)

    def mutate(self):
        mutation_prob_percent = self.mutation_prob * 100

        for individual in self.pop:
            for idx, gene in enumerate(individual.vec):
                if randint(1,100) <= mutation_prob_percent:
                    individual.vec[idx] = self._flip_gene(gene)
                    
    def _flip_gene(self, gene: int) -> int:
        assert ((gene == 1) or (gene == 0))

        if gene == 1:
            return 0
        return 1

    def keep_n_fittest(self, n):
        self.pop = self.pop[-n:]

    def get_crossover_vecs(self):
        crossover_vecs = []
        for ind in self.pop:
            crossover_vecs.append(ind.vec)
        return crossover_vecs

    def get_best(self):
        return self.pop[-1]

    def _mix_genetics(self, ind1: Individual, ind2: Individual, ratio: float = 0.5, 
                      keep_first: bool = True) -> list:
        """Mixes the genes of the two individuals.

        Args:
            ind1 (Individual): Individual that will be mixed with ind2.
            ind2 (Individual): Individual that will be mixed with ind1.
            ratio (float): What ratio should genes be mixed. ratio = .5 gives a 50/50 mix of genes.
            keep_first (Boolean): If true will keep gene from ind1 if lenght of vectors is odd.

        Returns:
            list: mixed 
        """
        if len(ind1) != len(ind2):
            # Raise meaningfull error
            raise Exception
        if keep_first:
            slice_idx = int(len(ind1) * ratio) + (len(ind1) * ratio> 0)
        else:
            slice_idx = int(len(ind1) * ratio)
        
        mixed = ind1[:slice_idx] + ind2[slice_idx:]
        return mixed


def main():
    from numpy import dot
    from numpy.linalg import norm  

    def scoring_func(vec):
        soln = [0, 1, 0, 1, 0]
        cos_sim = dot(soln, vec) / (norm(soln) * norm(vec))
        return cos_sim

    crossover_vecs = [
        [1, 0, 1, 0, 1],
        [1, 1, 1, 1, 1],
        [0, 1, 0, 1, 0]
    ]

    pop_size2 = 5
    n_genes2 = 5

    g2 = Generation(
        pop_size2, n_genes2, scoring_func, crossover_vecs=crossover_vecs
    )

    g2.sort_generation()
    print(g2.pop[-1].fitness)

if __name__ == "__main__":
    main()
