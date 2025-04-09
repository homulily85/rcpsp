from encoder.staircase import StaircaseSATEncoder


class StaircaseSCPBSATEncoder(StaircaseSATEncoder):
    def __init__(self, problem, makespan: int, timeout: int = None, enable_verify: bool = False):
        super().__init__(problem, makespan, timeout, enable_verify)

    def _SCPB_at_most_K(self, literals: list[int], weights: list[int], k: int):
        if len(literals) != len(weights):
            raise ValueError("The length of literals and weights must be the same.")
        always_false = []
        for w in weights:
            if w > k:
                self.sat_model.add_clause([-literals[weights.index(w)]])
                # print([-literals[weights.index(w)]])
                # always_false.append(weights.index(w))

        for i in range(len(always_false)):
            literals.remove(literals[always_false[i]])
            weights.remove(weights[always_false[i]])

        bit = {}
        for i in range(len(literals)):
            if i == 0:
                bit[literals[i]] = [self.sat_model.get_new_var() for _ in
                                    range(min(k, weights[i]))]
            elif i + 1 < k:
                bit[literals[i]] = [self.sat_model.get_new_var() for _ in
                                    range(
                                        min(k, sum([weights[j] for j in range(0, i + 1)])))]
            else:
                bit[literals[i]] = [self.sat_model.get_new_var() for _ in range(k)]

        # print(bit)

        for i in range(len(literals) - 1):
            for j in range(weights[i]):
                self.sat_model.add_clause([-literals[i], bit[literals[i]][j]])
                # print([-literals[i], bit[literals[i]][j]])

        for i in range(1, len(literals) - 1):
            for j in range(0, len(bit[literals[i - 1]])):
                self.sat_model.add_clause([-bit[literals[i - 1]][j], bit[literals[i]][j]])
                # print([-bit[literals[i - 1]][j], bit[literals[i]][j]])

        for i in range(1, len(literals) - 1):
            temp = []
            try:
                for j in range(0, len(bit[literals[i-1]])):
                    temp.append([-literals[i], -bit[literals[i - 1]][j],
                                 bit[literals[i]][j + weights[i]]])
                for c in temp:
                    self.sat_model.add_clause(c)
                    # print(c)
            except IndexError:
                continue

        for i in range(1, len(literals)):
            try:
                self.sat_model.add_clause(
                    [-literals[i], -bit[literals[i - 1]][k - weights[i]]])
                # print([-literals[i], -bit[literals[i - 1]][k - weights[i]]])
            except IndexError:
                continue

    def _resource_constraint(self):
        for t in range(6,self.makespan):
            for r in range(self.problem.nresources):
                # pb_constraint = PBConstr(self.sat_model, self.problem.capacities[r])
                literals = []
                weights = []
                for i in range(self.problem.njobs):
                    if t in range(self.ES[i], self.LC[i] + 1):
                        # pb_constraint.add_term(self.run[(i, t)], self.problem.requests[i][r])
                        literals.append(self.run[i,t])
                        weights.append(self.problem.requests[i][r])
                self._SCPB_at_most_K(literals,weights,self.problem.capacities[r])
                # pb_constraint.encode()

