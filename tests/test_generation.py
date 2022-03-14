import random
from src.generation import Generation

pop_size = 5
vec_len = 5

generation_no_crossover = Generation(pop_size, vec_len,
                                     lambda x: sum(x), [])
generation_3_crossover = Generation(pop_size, vec_len,
                                    lambda x: sum(x), [
    {
        "ind":0,
        "vec": [random.randint(0, 1) for _ in range(vec_len)],
        "fitness":1,
    },
    {
        "ind":1,
        "vec": [random.randint(0, 1) for _ in range(vec_len)],
        "fitness": 1,
    },
    {
        "ind":2,
        "vec": [random.randint(0, 1) for _ in range(vec_len)],
        "fitness": 1}
])


def test_vec_len():
    pop_1 = generation_no_crossover.pop
    for v in pop_1:
        assert len(v["vec"]) == vec_len

    pop_2 = generation_3_crossover.pop
    for v in pop_2:
        assert len(v["vec"]) == vec_len


def test_vec_vals():
    pop_1 = generation_no_crossover.pop
    for v in pop_1:
        assert sum(v["vec"]) <= vec_len

    pop_2 = generation_3_crossover.pop
    for v in pop_2:
        assert sum(v["vec"]) <= vec_len


def test_pop_idx():
    pop_1 = generation_no_crossover.pop
    sum_of_idx = 0

    for v in pop_1:
        sum_of_idx += v["idx"]

    assert sum_of_idx == (pop_size*4)/2

