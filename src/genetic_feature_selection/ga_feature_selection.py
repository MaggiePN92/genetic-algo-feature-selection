from generation import Generation
from tqdm import tqdm
from types import FunctionType


class GA:
    def __init__(self, k_generations: int, pop_size: int, n_genes: int,
                num_crossover_ind: int, score_func: FunctionType, crossover_vecs: list = [],
                 mutation_prob: float = 0.1) -> None:
        """Assigns initial values to important parameters. 

        Args:
            k_generations (int): How many runs/generations are to be made by the search.
            pop_size (int): The number of vectors/feature combinations in each generation.  
            n_genes (int): How many genes/features in each vector/feature combination.
            num_crossover_ind (int): How many vector/feature combinations are to be passed on to the next generation.
            score_func (FunctionType): The function that is used to evaluate the fitness of each vector/feature combination.
            crossover_vecs (list, optional): A list that contains vector/feature combinations from previous runs. Defaults to [].
            mutation_prob (float, optional): The probability that a gene is altered. Defaults to 0.1.
        """
        self.k_generations = k_generations
        self.pop_size = pop_size
        self.n_genes = n_genes
        self.num_crossover_ind = num_crossover_ind
        self.score_func = score_func
        self.crossover_vecs = crossover_vecs
        self.mutation_prob = mutation_prob
        
    def search(self) -> list:
        """Generate k generations. In each generation sort, mutate and
        keep n-fittest. Transfer n-fittest to next generation with 
        crossover_vecs.

        Returns:
            list: Best solution
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
