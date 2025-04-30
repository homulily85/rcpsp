from threading import Timer

from encoder.RCPSP_Encoder import RCPSPSolver
from encoder.problem import Problem
from encoder.sat.incremental_sat.SAT_model import SATModel


class SATSolver(RCPSPSolver):
    def __init__(self, problem: Problem, makespan: int, timeout: int = None,
                 enable_verify: bool = False):
        """Initialize the encoder with the problem, makespan, timeout, and verification flag."""
        super().__init__(problem, makespan, timeout, enable_verify)
        self.sat_model = SATModel()

        # Variable start_(i,t): boolean representing whether activity i starts at time t in STW(i)
        self.start: dict[tuple[int, int], int] = {}
        # Variable run_(i,t): boolean representing whether activity i is running at time t in RTW(i)
        self.run: dict[tuple[int, int], int] = {}

        self._preprocessing()

    def _preprocessing(self):
        self._calculate_time_windows()
        self._create_variable()

    def _create_variable(self):
        for i in range(self.problem.number_of_activities):
            for t in range(self.ES[i],
                           self.LS[i] + 1):  # t in STW(i) (start time window of activity i)
                self.sat_model.number_of_variable += 1
                self.start[(i, t)] = self.sat_model.number_of_variable

        for i in range(self.problem.number_of_activities):
            for t in range(self.ES[i],
                           self.LC[i] + 1):  # t in RTW(i) (run time window of activity i)
                self.sat_model.number_of_variable += 1
                self.run[(i, t)] = self.sat_model.number_of_variable

    def solve(self):
        assumptions = list(self.assumptions)

        self.sat_model.solver.clear_interrupt()
        timer = None  # Initialize timer

        if self.time_out is not None:
            def interrupt(s):
                s.interrupt()

            timer = Timer(self.time_out - self.time_used, interrupt, [self.sat_model.solver])
            timer.start()

        try:
            result = self.sat_model.solver.solve_limited(assumptions=assumptions,
                                                         expect_interrupt=True)
            self.time_used = self.sat_model.solver.time_accum()
            return result
        finally:
            if timer:
                timer.cancel()  # Cancel the timer if solve_limited finished early

    def decrease_makespan(self):
        """This method is used to decrease the makespan of the problem.
        It should be called after encode() and solve() methods.
        After calling this method, you will need to call solve() method to solve problem with new makespan."""
        self.get_solution()
        last_job = []
        for i in range(self.problem.number_of_activities):
            if self.problem.successors[i] == [self.problem.number_of_activities - 1]:
                last_job.append(i)

        actual_makespan = 0
        for job in last_job:
            if self.solution[job] + self.problem.durations[job] > actual_makespan:
                actual_makespan = self.solution[job] + self.problem.durations[job]

        if self.makespan == actual_makespan:
            self.assumptions.add(-self.start[self.problem.number_of_activities - 1, self.makespan])
            self.makespan-=1
        else:
            for m in range(actual_makespan + 1, self.makespan + 1):
                self.assumptions.add(-self.start[self.problem.number_of_activities - 1, m])
            self.makespan = actual_makespan
        self.solution = None

    def get_solution(self) -> list[int]:
        """Get the result of the problem where the result is a list of start times for each activity."""
        if self.sat_model.solver.get_model() is None:
            raise Exception(
                f"This problem is unsatisfiable with the current makespan {self.makespan}")

        if self.solution is not None:
            return self.solution

        sol = []
        for i in range(self.problem.number_of_activities):
            start_time_found = False
            for s in range(self.ES[i], self.LS[i] + 1):
                if self.sat_model.solver.get_model()[self.start[(i, s)] - 1] > 0:
                    sol.append(s)
                    start_time_found = True
                    break
            if not start_time_found:
                raise Exception(
                    f"Start time for activity {i} not found in the solution with makespan {self.makespan}"
                )
        self.solution = sol
        return sol
