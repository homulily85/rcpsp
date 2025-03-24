from encoder.PB_constr import PBConstr
from encoder.SAT_encoder import Encoder


class Thesis2022Encoder(Encoder):
    def encode(self):
        self._start_time_for_job_0()
        self._precedence_constraint()
        self._start_clause()
        self._consistency_constraint()
        self._redundant_constraint()
        self._resource_constraint()

    def _resource_constraint(self):
        for t in range(self.makespan):
            for r in range(self.problem.nresources):
                pb_constraint = PBConstr(self.sat_model, self.problem.capacities[r])
                for i in range(self.problem.njobs):
                    if t in range(self.ES[i], self.LC[i] + 1):
                        pb_constraint.add_term(self.x[(i, t)], self.problem.requests[i][r])
                pb_constraint.encode()

    def _redundant_constraint(self):
        for i in range(self.problem.njobs):
            for c in range(self.EC[i], self.LC[i]):
                self.sat_model.clauses.append(
                    [-self.x[(i, c)], self.x[(i, c + 1)],
                     self.y[(i, c - self.problem.durations[i] + 1)]])

    def _consistency_constraint(self):
        for i in range(self.problem.njobs):
            for s in range(self.ES[i], self.LS[i] + 1):  # s in STW(i)
                for t in range(s, s + self.problem.durations[i]):
                    self.sat_model.clauses.append(
                        [-self.y[(i, s)], self.x[(i, t)]])
                    self.sat_model.number_of_consistency_clause += 1

    def _start_clause(self):
        for i in range(1, self.problem.njobs):
            clause = []
            for s in range(self.ES[i], self.LS[i] + 1):  # s in STW(i)
                clause.append(self.y[(i, s)])
            self.sat_model.clauses.append(clause)

    def _precedence_constraint(self):
        for i in range(1, self.problem.njobs):
            for j in self.problem.predecessors[i]:
                for s in range(self.ES[i], self.LS[i] + 1):  # s in STW(i)
                    clause = [-self.y[(i, s)]]
                    # print(f'y{i}{s}', end=' ')

                    t = self.ES[j]
                    while t <= s - self.problem.durations[j] and t <= self.LS[j]:
                        clause.append(self.y[(j, t)])
                        # print(f'y{j}{t}', end=' ')
                        t += 1
                    # print()
                    self.sat_model.clauses.append(clause)

    def _start_time_for_job_0(self):
        self.sat_model.clauses.append([self.y[(0, 0)]])
