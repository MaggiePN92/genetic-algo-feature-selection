# import random
from src.generation import Generation
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
    for v in pop_1:
        assert len(v["vec"]) == n_genes


def test_n_genes():
    pop1 = g1.pop
    for v in pop1:
        assert len(v.vec) == n_genes

def test_sort_generation():
    g2._sort_generation()
    assert g2.pop[0] == [0, 1, 0, 1, 0]
    assert g2.pop[1] == [1, 1, 1, 1, 1]
    assert g2.pop[2] == [1, 0, 1, 0, 1]




# generation_no_crossover = Generation(pop_size, n_genes,
#                                      lambda x: sum(x), [])
# generation_3_crossover = Generation(pop_size, n_genes,
#                                     lambda x: sum(x), [
#     {
#         "ind":0,
#         "vec": [random.randint(0, 1) for _ in range(n_genes)],
#         "fitness":1,
#     },
#     {
#         "ind":1,
#         "vec": [random.randint(0, 1) for _ in range(n_genes)],
#         "fitness": 1,
#     },
#     {
#         "ind":2,
#         "vec": [random.randint(0, 1) for _ in range(n_genes)],
#         "fitness": 1}
# ])




#     pop_2 = generation_3_crossover.pop
#     for v in pop_2:
#         assert len(v["vec"]) == n_genes


# def test_vec_vals():
#     pop_1 = generation_no_crossover.pop
#     for v in pop_1:
#         assert sum(v["vec"]) <= n_genes

#     pop_2 = generation_3_crossover.pop
#     for v in pop_2:
#         assert sum(v["vec"]) <= n_genes


# def test_pop_idx():
#     pop_1 = generation_no_crossover.pop
#     sum_of_idx = 0

#     for v in pop_1:
#         sum_of_idx += v["idx"]

#     assert sum_of_idx == (pop_size*4)/2
