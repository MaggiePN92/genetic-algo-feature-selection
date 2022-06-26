from cmath import nan
import math
from generation import Generation, mix_genetics
from tqdm import tqdm
from typing import List, Tuple, Callable
import numpy as np 
from genetic_feature_selection.individual import Individual


class GA:
    def __init__(self, k_generations: int, pop_size: int, n_genes: int,
                num_crossover_ind: int, score_func: Callable, crossover_vecs: list = [],
                 mutation_prob: float = 0.1, num_parents_mating: int = 2) -> None:
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
        self.num_parents_mating = num_parents_mating
        self.best_soln = Individual(None, 0)
        
    def search(self) -> list:
        """
        1. generate initial population.
        Untill k_generations is reached: 
            2. sort generation by fitness:
            3. keep the best num_crossover_ind.
            3. mate the remaining individuals, their offspring replaces 
            the individuals that were removed in step 3.
            4. repeat from step 2.

        Returns:
            list: Best solution
        """
        
        # generating/setting initial population 
        crossover_vecs = self.crossover_vecs
        g = Generation(self.pop_size, self.n_genes, self.score_func,
                       crossover_vecs, self.mutation_prob)
        g.init_pop()


        for gen in tqdm(range(self.k_generations)):
            # Problem: population is not initialized second iteration
            g.mutate()
            g.sort_generation()
            g.keep_n_fittest(self.num_crossover_ind)
            self.best_soln = g.get_best() if g.get_best().fitness > self.best_soln.fitness else self.best_soln

            # select pairs of parents
            mating_pairs = self.select_mating_pairs(g.pop, self.num_parents_mating)
            offsprings = self.mate_pairs(mating_pairs)

            crossover_vecs = g.get_vecs()
            crossover_vecs.extend(offsprings)

            g = Generation(self.pop_size, self.n_genes, self.score_func,
                          crossover_vecs, self.mutation_prob)
            g.init_pop()
            
        return self.best_soln

    def mate_pairs(self, mating_pairs: List[Tuple[int, int]]) -> List[int]:
        offsprings = []
        for pair in mating_pairs:
            offspring_vec = mix_genetics(pair[0].vec, pair[1].vec)
            offsprings.append(offspring_vec)
        return offsprings

    def select_mating_pairs(self, parents: List[Individual], num_parents_mating: int) -> List[Tuple[Individual, Individual]]:
        """ Use fitness score to calculate the probability that an individual
        is chosen to be mated.  
        """

        # Let's say fitness for individual 1,2,3 and 4 is given by
        # [0.71, 0.76, 0.8, 0.83]. tot_fitness is then 3,1. I then 
        # want to select individual 1 if random_number is <= 0.71/3.1. 
        # Individual 2 is selected if random_number is > 0.71/3.1 and less
        # or equal to (0.76 + 0.71)/3.1, and so on.

        fitness = np.array([ind.fitness for ind in parents])
        fitness[np.isnan(fitness)] = 0
        tot_fitness = np.sum(fitness)
        probs = fitness / tot_fitness
        
        mates = []
        # consider if num_parents_mating should be number of offspring produced
        for _ in range(num_parents_mating//2):
            parent1_idx = np.random.choice(len(parents), p=probs)
            parent2_idx = np.random.choice(len(parents), p=probs)
            mates.append((parents[parent1_idx], parents[parent2_idx]))
        return mates

        

        

            




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
