from enum import Enum, auto


class SOLVER_STATUS(Enum):
    OPTIMUM = auto()
    UNSATISFIABLE = auto()
    SATISFIABLE = auto()
    UNKNOWN = auto()