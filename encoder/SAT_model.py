from enum import Enum

from pysat.solvers import Glucose4


class NUMBER_OF_LITERAL(Enum):
    ZERO = 0
    ONE = 1
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE_TO_TEN = 5
    MORE_THAN_TEN = 11


class SATModel:
    def __init__(self):
        self.number_of_variable = 0
        self.number_of_clause = 0
        self.number_of_PB_clause = 0
        self.number_of_consistency_clause = 0
        self.number_of_literal = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 11: 0}
        self.solver = Glucose4(use_timer=True, incr=True)

    def get_new_var(self) -> int:
        self.number_of_variable += 1
        return self.number_of_variable

    def add_clause(self, clause: list[int]) -> None:
        self.solver.add_clause(clause)
        self.number_of_clause += 1
        if len(clause) > 10:
            self.number_of_literal[NUMBER_OF_LITERAL.MORE_THAN_TEN.value] += 1
        elif len(clause) > 4:
            self.number_of_literal[NUMBER_OF_LITERAL.FIVE_TO_TEN.value] += 1
        else:
            self.number_of_literal[len(clause)] += 1
