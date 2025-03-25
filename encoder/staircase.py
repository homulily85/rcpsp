from encoder.PB_constr import PBConstr
from encoder.SAT_encoder import Encoder


class StaircaseEncoder(Encoder):
    def encode(self):
        self._resource_constraint()
        self._start_time_for_job_0()
        self._precedence_constraint()
        self._consistency_constraint()
        # Add redundant clauses that should improve runtime
        self._redundant_constraint()

    def _precedence_constraint(self):
        for j in range(1, self.problem.njobs):
            for i in self.problem.successors[j]:
                m = max(self.ES[j] + self.problem.durations[j], self.ES[i])

                temp: list[int] = []

                for k in range(m, self.LS[i] + 1):
                    # Linking subset to auxiliary variable
                    temp.append(self.y[i, k])
                    current = tuple(temp)
                    if current not in self.register:
                        self.register[current] = self.sat_model.get_new_var()

                        # print(f'{current}->{self.register[current]}')
                        # for x in range(m, k + 1):
                        #     print(f'y{i}{x}', end=' ')
                        # print(f'-->{self.register[current]}')

                        # Constraint for staircase
                        self.sat_model.add_clause([-self.y[i, k], self.register[current]])
                        # print(f'-y{i}{k} {self.register[current]}')
                        if k != m:
                            previous = tuple(temp[:len(temp) - 1])
                            self.sat_model.add_clause(
                                [-self.register[previous], self.register[current]])
                            # print(f'-{self.register[previous]} {self.register[current]}')
                            self.sat_model.add_clause(
                                [self.register[previous], self.y[i, k], -self.register[current]])
                            # print(f'{self.register[previous]} y{i}{k} -{self.register[current]}')
                            self.sat_model.add_clause([-self.y[i, k], -self.register[previous]])
                            # print(f'-y{i}{k} -{self.register[previous]}')

                if len([self.y[i, t] for t in range(m, self.LS[i] + 1)]) == 1:
                    self.sat_model.add_clause(
                        [self.y[i, t] for t in range(m, self.LS[i] + 1)]
                    )
                else:
                    self.sat_model.add_clause(
                        [self.register[
                             tuple([self.y[i, t] for t in range(m, self.LS[i] + 1)])
                         ]]
                    )

                # print(self.register[
                #           tuple([self.y[i, t] for t in range(m, self.LS[i] + 1)])
                #       ])

                temp = []
                for k in range(self.LS[j], self.ES[j] - 1, -1):
                    temp.append(self.y[j, k])
                    current = tuple(temp)
                    if current not in self.register:
                        self.register[current] = self.sat_model.get_new_var()

                        # for x in range(self.LS[j], k - 1, -1):
                        # print(f'y{j}{x}', end=' ')
                        # print(f'-->{self.register[current]}')

                        self.sat_model.add_clause([-self.y[j, k], self.register[current]])
                        # print(f'-y{j}{k} {self.register[current]}')

                        if k != self.LS[j]:
                            previous = tuple(temp[:len(temp) - 1])
                            self.sat_model.add_clause(
                                [-self.register[previous], self.register[current]])
                            # print(f'-{self.register[previous]} {self.register[current]}')
                            self.sat_model.add_clause(
                                [self.register[previous], self.y[j, k], -self.register[current]])
                            # print(f'{self.register[previous]} y{j}{k} -{self.register[current]}')
                            self.sat_model.add_clause([-self.y[j, k], -self.register[previous]])
                            # print(f'-y{j}{k} -{self.register[previous]}')

                if len(tuple([self.y[j, t] for t in range(self.LS[j], self.ES[j] - 1, -1)])) == 1:
                    self.sat_model.add_clause([
                        self.y[j, t] for t in range(self.ES[j], self.LS[j] + 1)
                    ])
                else:
                    self.sat_model.add_clause(
                        [self.register[
                             tuple([self.y[j, t] for t in range(self.LS[j], self.ES[j] - 1, -1)])]]
                    )

                # print(self.register[
                #           tuple([self.y[j, t] for t in range(self.LS[j], self.ES[j] - 1, -1)])])

                for t in range(m, self.LS[i] + 1):
                    first_half_temp = list(self.y[j, k] for k in
                                           range(t - self.problem.durations[j] + 1, self.LS[j] + 1))

                    if len(first_half_temp) == 0:
                        continue

                    first_half_temp.reverse()
                    first_half = tuple(first_half_temp)

                    second_half = tuple(self.y[i, k] for k in range(m, t + 1))

                    self.sat_model.add_clause(
                        [-self.register[first_half], -self.register[second_half]])

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
                self.sat_model.add_clause(
                    [-self.x[(i, c)], self.x[(i, c + 1)],
                     self.y[(i, c - self.problem.durations[i] + 1)]])

    def _consistency_constraint(self):
        for i in range(self.problem.njobs):
            for s in range(self.ES[i], self.LS[i] + 1):  # s in STW(i)
                for t in range(s, s + self.problem.durations[i]):
                    self.sat_model.add_clause(
                        [-self.y[(i, s)], self.x[(i, t)]])
                    self.sat_model.number_of_consistency_clause += 1

    def _start_time_for_job_0(self):
        self.sat_model.add_clause([self.y[(0, 0)]])
