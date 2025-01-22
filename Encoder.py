from pysat.solvers import Glucose3

from PBConstr import PBConstr
from Problem import Problem
from SATModel import SATModel


class Encoder:
    def __init__(self, problem: Problem, bounds: tuple[int, int]):
        self.sat_model = SATModel()
        self.problem = problem
        # The lower and upper bounds for the makespan that are currently being used
        self.LB = bounds[0]
        self.UB = bounds[1]
        # For each activity: earliest start, earliest close, latest start, and latest close time
        self.ES = None  # Initialize latter
        self.EC = [0] * self.problem.njobs
        self.LS = [self.UB] * self.problem.njobs
        self.LC = None  # Initialize latter

        # Variable y_(i,t): boolean representing whether activity i starts at time t in STW(i)
        self.y = [[] for _ in range(self.problem.njobs)]
        # Variable x_(i,t): boolean representing whether activity i is running at time t in RTW(i)
        self.x = [[] for _ in range(self.problem.njobs)]

        self.preprocessFeasible = False

        self.initialise()

    def initialise(self):
        self.preprocessFeasible = self.calc_time_windows()

        if self.preprocessFeasible:
            self.create_variable()

        else:
            print('Preprocessing found instance to be infeasible')
            exit(1)

    def create_variable(self):
        for i in range(self.problem.njobs):
            for t in range(self.ES[i],
                           self.LS[i] + 1):  # t in STW(i) (start time window of activity i)
                self.sat_model.nvariable += 1
                self.y[i].append(self.sat_model.nvariable)

        for i in range(self.problem.njobs):
            for t in range(self.ES[i],
                           self.LC[i] + 1):  # t in RTW(i) (run time window of activity i)
                self.sat_model.nvariable += 1
                self.x[i].append(self.sat_model.nvariable)

    def calc_time_windows(self) -> bool:
        # Use a queue for breadth-first traversal of the precedence graph
        queue = [0]

        while len(queue) > 0:
            job = queue.pop(0)
            duration = self.problem.durations[job]
            # Move finish until it is feasible considering resource constraints
            feasible_final = False

            while not feasible_final:
                feasible = True

                for k in range(self.problem.nresources):
                    if not feasible:
                        break

                    for t in range(duration - 1, -1, -1):
                        if not feasible:
                            break
                        if self.problem.requests[job][k][t] > self.problem.capacities[k][
                            self.EC[job] - duration + t]:
                            feasible = False
                            self.EC[job] += 1

                if feasible:
                    feasible_final = True

                if self.EC[job] > self.UB:
                    return False

            # Update finish times, and enqueue successors
            for succ in self.problem.successors[job]:
                c = self.EC[job] + self.problem.durations[succ]
                if self.EC[succ] < c:
                    self.EC[
                        succ] = c  # Use maximum values, because we are interested in critical paths
                queue.append(succ)

        # Calculate latest feasible start times
        queue.append(self.problem.njobs - 1)
        while len(queue) > 0:
            job = queue.pop(0)
            duration = self.problem.durations[job]

            # Move start until it is feasible considering resource constraints
            feasible_final = False

            while not feasible_final:
                feasible = True
                for k in range(self.problem.nresources):
                    if not feasible:
                        break
                    for t in range(duration):
                        if not feasible:
                            break
                        if self.problem.requests[job][k][t] > self.problem.capacities[k][
                            self.LS[job] + t]:
                            feasible = False
                            self.LS[job] -= 1

                if feasible:
                    feasible_final = True

                if self.LS[job] < 0:
                    return False

            # Update start times, and enqueue predecessors
            for pred in self.problem.predecessors[job]:
                s = self.LS[job] - self.problem.durations[pred]
                if s < self.LS[pred]:
                    self.LS[pred] = s  # Use minimum values for determining critical paths
                queue.append(pred)

        self.ES = [self.EC[i] - self.problem.durations[i] for i in range(self.problem.njobs)]
        self.LC = [self.LS[i] + self.problem.durations[i] for i in range(self.problem.njobs)]

        return True

    def encode(self):
        # Job 0 starts at 0 (10)
        self.sat_model.clauses.append([self.y[0][0]])

        # Precedence clauses (11)
        for i in range(1, self.problem.njobs):
            for j in self.problem.predecessors[i]:
                for s in range(self.ES[i], self.LS[i] + 1):  # s in STW(i)
                    clause = [-self.y[i][-self.ES[i] + s]]  # -self.ES[i] to convert to list index

                    t = self.ES[j]
                    while t <= s - self.problem.durations[j] and t <= self.LS[j]:
                        clause.append(self.y[j][-self.ES[j] + t])
                        t += 1

                    self.sat_model.clauses.append(clause)

        # Start clauses (12)
        for i in range(1, self.problem.njobs):
            clause = []
            for s in range(self.ES[i], self.LS[i] + 1):  # s in STW(i)
                clause.append(self.y[i][-self.ES[i] + s])
            self.sat_model.clauses.append(clause)

        # Consistency clauses (13)
        for i in range(self.problem.njobs):
            for s in range(self.ES[i], self.LS[i] + 1):  # s in STW(i)
                for t in range(s, s + self.problem.durations[i]):
                    self.sat_model.clauses.append(
                        [-self.y[i][-self.ES[i] + s], self.x[i][-self.ES[i] + t]])

        # Add redundant clauses that should improve runtime (14)
        for i in range(self.problem.njobs):
            for c in range(self.EC[i], self.LC[i]):
                self.sat_model.clauses.append(
                    [-self.x[i][-self.ES[i] + c], self.x[i][-self.ES[i] + c + 1],
                     self.y[i][-self.ES[i] + c - self.problem.durations[i] + 1]])

        # Add resource constraints(15)
        for k in range(self.problem.nresources):
            for t in range(self.UB):
                constraint = PBConstr(self.sat_model, self.problem.capacities[k][t])
                for i in range(self.problem.njobs):
                    if t < self.ES[i] or t >= self.LC[i]:
                        continue
                    for e in range(self.problem.durations[i]):
                        if t - e < self.ES[i] or t - e > self.LS[i]:
                            continue
                        q = self.problem.requests[i][k][e]
                        if q == 0:
                            continue
                        constraint.add_term(self.y[i][-self.ES[i] + t - e], q)
                if constraint.number_of_term() == 0:
                    continue
                constraint.encode()

    def get_result(self, model: list[int]) -> list[int]:
        sol = []
        for i in range(self.problem.njobs):
            for s in range(self.ES[i], self.LS[i] + 1):
                if model[self.y[i][-self.ES[i] + s] - 1] > 0:
                    sol.append(s)
                    break
        return sol
