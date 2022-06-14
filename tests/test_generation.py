# import random
from genetic_feature_selection.generation import Generation
from genetic_feature_selection.individual import Individual
from numpy import dot
from numpy.linalg import norm

#### FIXTURE #####


def scoring_func(vec):
    soln = [0, 1, 0, 1, 0]
    cos_sim = dot(soln, vec) / (norm(soln) * norm(vec))
    return cos_sim

pop_size = 8
n_genes = 5

#### WITHOUT INITAL POP / CROSSOVER VECS ####
g1 = Generation(
    pop_size, n_genes, scoring_func
)

#### WITH INITAL POP / CROSSOVER VECS ####
crossover_vecs = [
    [1, 0, 1, 0, 1],
    [1, 1, 1, 1, 1],
    [0, 1, 0, 1, 0]
]

pop_size2 = 3
n_genes2 = 5

g2 = Generation(
    pop_size2, n_genes2, scoring_func, crossover_vecs=crossover_vecs
)
#####################


def test_n_genes():
    pop1 = g1.pop
    for v in pop1:
        assert len(v.vec) == n_genes

def test_sort_generation():
    g2.sort_generation()
    assert g2.pop[-1].vec == [0, 1, 0, 1, 0]

def test_keep_n_fittest():
    g3 = Generation(
        8, 5, scoring_func
    )

    g3.keep_n_fittest(n = 6)
    assert len(g3.pop) == 6
    g3.keep_n_fittest(n = 4)
    assert len(g3.pop) == 4

def test_mutation():
    pass 

"""def test_crossover():
    g_crossover = Generation(
        2, 5, scoring_func, 
        crossover_vecs=[[1,1,1,1,1], [0,0,0,0,0]]
    )
    g_crossover.mutate()
    crossed_over = g_crossover.get_crossover_vecs()

    assert crossover_vecs[0] != crossed_over[0]
    assert crossover_vecs[0] != crossed_over[1]
    assert crossover_vecs[1] != crossed_over[0]
    assert crossover_vecs[1] != crossed_over[1]"""


def test_mix_genetics():
    g = Generation(
        2, 5, scoring_func, 
        crossover_vecs=[[1,1,1,1,1], [0,0,0,0,0]]
    )
    mixed = g._mix_genetics([1,1,1,1,1], [0,0,0,0,0])
    assert mixed == [1,1,1,0,0]
