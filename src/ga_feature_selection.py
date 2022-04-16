
from generation import Generation
from tqdm import tqdm


class GA:
    def __init__(self, k_generations, pop_size, n_genes,
                num_crossover_ind, score_func, crossover_vecs = [],
                 mutation_prob = 0.1):
        self.k_generations = k_generations
        self.pop_size = pop_size
        self.n_genes = n_genes
        self.num_crossover_ind = num_crossover_ind
        self.score_func = score_func
        self.crossover_vecs = crossover_vecs
        self.mutation_prob = mutation_prob
        
    def search(self):
        """ Generate k generations. In each generation sort, mutate and
        keep n-fittest. Transfer n-fittest to next generation with 
        crossover_vecs.
        
        :returns best solution 
        """
        crossover_vecs = self.crossover_vecs

        for gen in tqdm(range(self.k_generations)):
            g = Generation(self.pop_size, self.n_genes, self.score_func,
                          crossover_vecs, self.mutation_prob)
            
            g.mutate()
            g.sort_generation()
            g.keep_n_fittest(self.num_crossover_ind)
            crossover_vecs = g.get_crossover_vecs()
        
        return g.get_best()


def main():
    from math import hypot
    from numpy import dot
    from numpy.linalg import norm

    def scoring_func(vec):
        soln = [0, 1, 0, 1, 0]
        cos_sim = dot(soln, vec) / (norm(soln) * norm(vec))
        return cos_sim
    
    ga_search = GA(1000, 10, 5, 4, scoring_func)
    best_result = ga_search.search()

    print(f"Best score: {scoring_func(best_result.vec)}")


if __name__ == "__main__":
    main()
