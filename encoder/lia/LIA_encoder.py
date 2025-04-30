from z3 import Int, Bool, And, Implies, sat, unsat, If

from encoder.RCPSP_Encoder import RCPSPEncoder
from encoder.lia.LIA_model import LIAModel


class LIAEncoder(RCPSPEncoder):
    def __init__(self, problem, makespan, timeout=None, enable_verify=False):
        super().__init__(problem, makespan, timeout, enable_verify)
        self.lia_model = LIAModel()

        # Start Time Variables for Each Job
        self.start = {}
        # Boolean Variables for Activity Execution
        self.run = {}

        self._preprocessing()
        self.shortest_paths = None

    def _preprocessing(self):
        self._calculate_time_windows()
        self._create_variable()

    def _create_variable(self):
        self.start = {job: Int(f"start_{job}") for job in range(self.problem.njobs)}
        self.run = {
            (i, t): Bool(f"X_{i}_{t}")
            for i in range(self.problem.njobs)
            for t in range(self.ES[i], self.LC[i] + 1)
        }

    def encode(self):
        # First Job Starts at 0
        self.lia_model.solver.add(self.start[0] == 0)

        # Ensure jobs start within valid time windows
        for i in range(1, self.problem.njobs):
            self.lia_model.solver.add(self.start[i] >= self.ES[i])
            self.lia_model.solver.add(self.start[i] <= self.LS[i])

        # Enforce the precedences
        for i in range(self.problem.njobs):
            for j in self.problem.successors[i]:
                self.lia_model.solver.add(
                    self.start[j] >= self.start[i] + self.problem.durations[i])

        # Consistency Constraints
        for i in range(self.problem.njobs):
            for t in range(self.ES[i], self.LC[i] + 1):
                self.lia_model.solver.add(Implies(self.run[i, t], And(self.start[i] <= t,
                                                                      t < self.start[i] +
                                                                      self.problem.durations[i])))

        # Resource Constraints (Optimized for LIA)
        for k in range(self.problem.nresources):
            for t in range(self.makespan + 1):
                active_jobs = [
                    If(And(self.start[i] <= t, t < self.start[i] + self.problem.durations[i]),
                       self.problem.requests[i][k], 0)
                    for i in range(self.problem.njobs)]
                self.lia_model.solver.add(
                    sum(active_jobs) <= self.problem.capacities[k])  # LIA encoding

    def solve(self):
        assumptions = list(self.assumptions)
        if self.time_out is not None:
            self.lia_model.solver.set("timeout", self.time_out * 1000 - int(self.time_used * 1000))

        result = self.lia_model.solver.check(assumptions)

        for k, v in self.lia_model.solver.statistics():
            if k == "time":
                self.time_used += v
                break

        if result == sat:
            return True
        elif result == unsat:
            return False
        else:
            return None

    def decrease_makespan(self):
        """This method is used to decrease the makespan of the problem.
        It should be called after encode() and solve() methods.
        After calling this method, you will need to call solve() method to solve problem with new makespan."""
        self.get_solution()
        last_job = []
        for i in range(self.problem.njobs):
            if self.problem.successors[i] == [self.problem.njobs - 1]:
                last_job.append(i)

        actual_makespan = 0
        for job in last_job:
            if self.solution[job] + self.problem.durations[job] > actual_makespan:
                actual_makespan = self.solution[job] + self.problem.durations[job]

        if self.makespan == actual_makespan:
            self.assumptions.add(self.start[self.problem.njobs - 1] != self.makespan)
            self.makespan -= 1
        else:
            for m in range(actual_makespan + 1, self.makespan + 1):
                self.assumptions.add(self.start[self.problem.njobs - 1] != m)
            self.makespan = actual_makespan

        self.solution = None

    def get_solution(self) -> list[int]:
        """Get the result of the problem where the result is a list of start times for each activity."""
        if self.solution is None:
            self.solution = [self.lia_model.solver.model()[self.start[i]].as_long() for i in
                             range(self.problem.njobs)]

        return self.solution
