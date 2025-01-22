from SATModel import SATModel
from pysat.pb import *


class PBConstr:
    def __init__(self, model: SATModel, bound: int):
        self.lits = []
        self.weights = []
        self.bound = bound
        self.model = model
        self.var_index = {}

    def add_term(self, lit: int, weights: int):
        self.lits.append(lit)
        self.weights.append(weights)

    def encode(self):
        cnf = PBEnc.leq(lits=self.lits, weights=self.weights, bound=self.bound,
                        top_id=self.model.nvariable).clauses

        if not cnf:
            return

        max = -1
        for clause in cnf:
            for var in clause:
                if abs(var) > max:
                    max = abs(var)

        if max == -1:
            return

        self.model.nvariable = max
        self.model.clauses.extend(cnf)

    def number_of_term(self) -> int:
        return len(self.lits)
