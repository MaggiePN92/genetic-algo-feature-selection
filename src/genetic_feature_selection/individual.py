from dataclasses import dataclass
from typing import List


@dataclass
class Individual:
    vec: List[int]
    fitness: int
