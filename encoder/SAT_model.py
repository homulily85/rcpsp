class SATModel:
    def __init__(self):
        self.number_of_variable = 0
        self.clauses: list[list[int]] = []
        self.number_of_PB_clause = 0
        self.number_of_consistency_clause = 0

    def get_new_var(self) -> int:
        self.number_of_variable += 1
        return self.number_of_variable

    def number_of_literal(self) -> dict[int, int]:
        self.number_of_literal = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 11: 0}

        for clause in self.clauses:
            t = len(clause)
            if t > 10:
                self.number_of_literal[11] += 1
            elif t > 4:
                self.number_of_literal[5] += 1
            else:
                self.number_of_literal[t] += 1

        return self.number_of_literal
