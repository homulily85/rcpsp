from pysat.pb import *

from encoder.sat.max_sat.max_sat_model import MaxSATModel


class PBConstraint:
    def __init__(self, model: MaxSATModel, bound: int):
        self._literals = []
        self._weights = []
        self._bound = bound
        self._model = model

    def add_term(self, lit: int, weights: int):
        self._literals.append(lit)
        self._weights.append(weights)

    def encode(self):
        cnf = PBEnc.leq(lits=self._literals, weights=self._weights, bound=self._bound,
                        top_id=self._model.number_of_variable, encoding=EncType.bdd).clauses

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
        for clause in cnf:
            self._model.add_hard_clause(clause)

        self._model.number_of_PB_clause += len(cnf)
