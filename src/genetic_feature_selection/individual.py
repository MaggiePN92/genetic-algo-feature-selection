from dataclasses import dataclass
from typing import List


@dataclass
class Individual:
    id: int
    vec: List[List]
    fitness: int
