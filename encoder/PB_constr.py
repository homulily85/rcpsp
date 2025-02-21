from encoder.SAT_model import SATModel
from pysat.pb import *


class PBConstr:
    def __init__(self, model: SATModel, bound: int):
        self._literals = []
        self._weights = []
        self._bound = bound
        self._model = model
        self._var_index = {}

    def add_term(self, lit: int, weights: int):
        self._literals.append(lit)
        self._weights.append(weights)

    def encode(self):
        cnf = PBEnc.leq(lits=self._literals, weights=self._weights, bound=self._bound,
                        top_id=self._model.number_of_variable).clauses

        # cnf can be empty
        if not cnf:
            return

        M = -1
        for clause in cnf:
            for var in clause:
                if abs(var) > M:
                    M = abs(var)

        if M == -1:
            return

        self._model.number_of_variable = max(M, self._model.number_of_variable)
        self._model.clauses.extend(cnf)

    # def number_of_term(self) -> int:
    #     return len(self._literals)
