from queue import Queue

from encoder.PB_constr import PBConstr
from encoder.problem import Problem
from encoder.SAT_model import SATModel


class PreprocessingFailed(Exception):
    pass


class Encoder:
    def __init__(self, problem: Problem, makespan: int):
        self.sat_model = SATModel()
        self.problem = problem
        self.makespan = makespan
        # For each activity: earliest start, earliest close, latest start, and latest close time
        self.ES = [0] * self.problem.njobs
        self.EC = [0] * self.problem.njobs
        self.LS = [self.makespan] * self.problem.njobs
        self.LC = [self.makespan] * self.problem.njobs

        # Variable y_(i,t): boolean representing whether activity i starts at time t in STW(i)
        self.y: dict[tuple[int, int], int] = {}
        # Variable x_(i,t): boolean representing whether activity i is running at time t in RTW(i)
        self.x: dict[tuple[int, int], int] = {}

        self._preprocessing()

    def _preprocessing(self):
        self._calc_time_windows()
        self._create_variable()

    def _create_variable(self):
        for i in range(self.problem.njobs):
            for t in range(self.ES[i],
                           self.LS[i] + 1):  # t in STW(i) (start time window of activity i)
                self.sat_model.number_of_variable += 1
                self.y[(i, t)] = self.sat_model.number_of_variable

        for i in range(self.problem.njobs):
            for t in range(self.ES[i],
                           self.LC[i] + 1):  # t in RTW(i) (run time window of activity i)
                self.sat_model.number_of_variable += 1
                self.x[(i, t)] = self.sat_model.number_of_variable

    def _calc_time_windows(self):
        # Calculate ES and EC
        mark = [False] * self.problem.njobs
        queue = Queue()
        queue.put(0)

        while not queue.empty():
            curr_job = queue.get()
            mark[curr_job] = True

            self.ES[curr_job] = 0 if len(self.problem.predecessors[curr_job]) == 0 else max(
                self.EC[p] for p in self.problem.predecessors[curr_job])

            self.EC[curr_job] = self.ES[curr_job] + self.problem.durations[curr_job]
            if self.EC[curr_job] > self.makespan:
                raise PreprocessingFailed

            for successor in self.problem.successors[curr_job]:
                if not mark[successor]:
                    queue.put(successor)

        # Calculate LC and LS
        mark = [False] * self.problem.njobs
        queue = Queue()
        queue.put(self.problem.njobs - 1)

        while not queue.empty():
            curr_job = queue.get()
            mark[curr_job] = True

            self.LC[curr_job] = self.makespan if len(
                self.problem.successors[curr_job]) == 0 else min(
                self.LS[s] for s in self.problem.successors[curr_job])

            self.LS[curr_job] = self.LC[curr_job] - self.problem.durations[curr_job]
            if self.LS[curr_job] < 0:
                raise PreprocessingFailed

            for predecessor in self.problem.predecessors[curr_job]:
                if not mark[predecessor]:
                    queue.put(predecessor)

    def encode(self):
        # Job 0 starts at 0 (10)
        self.sat_model.clauses.extend(self._encode_constraint_10())

        self.sat_model.clauses.extend(self._encode_new_precedence_constraint())

        # Consistency clauses (13)
        self.sat_model.clauses.extend(self._encode_constraint_13())

        # Add redundant clauses that should improve runtime (14)
        self.sat_model.clauses.extend(self._encode_constraint_14())

        # Add resource constraints(15)
        self._encode_constraint_15()

    @staticmethod
    def _AMO_binomial(var: list[int]):
        if len(var) == 0:
            return []
        elif len(var) == 1:
            return [[var[0]]]

        clauses = []
        for i in range(len(var)):
            for j in range(i + 1, len(var)):
                clauses.append([-var[i], -var[j]])
        return clauses

    @staticmethod
    def _ALO_binomial(var: list[int]):
        return [var]

    @staticmethod
    def _EXO(var: list[int]):

        t = Encoder._AMO_binomial(var)
        t.extend((Encoder._ALO_binomial(var)))
        return t

    @staticmethod
    def _AMZ(var: list[int]):
        return [[-i] for i in var]

    def _encode_new_precedence_constraint(self):
        clauses = []

        for j in range(1, self.problem.njobs):
            for i in self.problem.successors[j]:
                # for t in range(self.ES[j], self.LS[j] + 1):
                #     print(f'y{j}{t}', end=' ')
                # print('=1')
                clauses.extend(Encoder._EXO([
                    self.y[j, t] for t in range(self.ES[j], self.LS[j] + 1)
                ]))

                amz = [self.y[i, t] for t in
                       range(self.ES[i], self.ES[j] + self.problem.durations[j])
                       ]
                if len(amz) > 0:
                    clauses.extend(Encoder._AMZ(amz))

                # for t in range(self.ES[i], self.ES[j] + self.problem.durations[j]):
                #     print(f'y{i}{t}', end=' ')
                # print('=0')

                x = max(self.ES[j] + self.problem.durations[j], self.ES[i])

                for t in range(x, self.LS[i] + 1):
                    temp = [self.y[j, k] for k in
                            range(t - self.problem.durations[j] + 1, self.LS[j] + 1)]
                    if len(temp) == 0:
                        continue

                    # for k in range(x, t + 1):
                    #     print(f'y{i}{k}', end=' ')
                    temp.extend([self.y[i, k] for k in range(x, t + 1)])
                    # for k in range(t - self.problem.durations[j] + 1, self.LS[j] + 1):
                    #     print(f'y{j}{k}', end=' ')
                    # print('<=1')
                    clauses.extend(Encoder._AMO_binomial(temp))

                clauses.extend(Encoder._EXO(
                    [self.y[i, t] for t in range(self.ES[i], self.LS[i] + 1)]
                ))
                # for t in range(self.ES[i], self.LS[i] + 1):
                #     print(f'y{i}{t}', end=' ')
                # print('=1')

        return clauses

    def _encode_constraint_15(self):
        for t in range(self.makespan):
            for r in range(self.problem.nresources):
                pb_constraint = PBConstr(self.sat_model, self.problem.capacities[r])
                for i in range(self.problem.njobs):
                    if t in range(self.ES[i], self.LC[i] + 1):
                        pb_constraint.add_term(self.x[(i, t)], self.problem.requests[i][r])
                pb_constraint.encode()

    def _encode_constraint_14(self) -> list[list[int]]:
        clauses = []
        for i in range(self.problem.njobs):
            for c in range(self.EC[i], self.LC[i]):
                clauses.append(
                    [-self.x[(i, c)], self.x[(i, c + 1)],
                     self.y[(i, c - self.problem.durations[i] + 1)]])

        return clauses

    def _encode_constraint_13(self) -> list[list[int]]:
        clauses = []
        for i in range(self.problem.njobs):
            for s in range(self.ES[i], self.LS[i] + 1):  # s in STW(i)
                for t in range(s, s + self.problem.durations[i]):
                    clauses.append(
                        [-self.y[(i, s)], self.x[(i, t)]])

        return clauses

    def _encode_constraint_10(self) -> list[list[int]]:
        return [[self.y[(0, 0)]]]

    def get_result(self, model: list[int]) -> list[int]:
        sol = []
        for i in range(self.problem.njobs):
            for s in range(self.ES[i], self.LS[i] + 1):
                if model[self.y[(i, s)] - 1] > 0:
                    sol.append(s)
                    break
        return sol