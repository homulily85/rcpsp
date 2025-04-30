from threading import Timer

from encoder.model.sat_model import IncrementalSATModel
from encoder.sat.sat_solver import SATSolver


class IncrementalSATSolver(SATSolver):
    """
    Base class for solving the Resource-Constrained Project Scheduling Problem (RCPSP) using incremental SAT method.
    """

    def __init__(self, input_file: str,
                 lower_bound: int,
                 upper_bound: int,
                 timeout: int = None,
                 enable_verify: bool = False):
        """Initialize the encoder with the problem, makespan, timeout, and verification flag."""
        self._sat_model = IncrementalSATModel()
        self._makespan = upper_bound
        self._assumptions = set()
        super().__init__(input_file, lower_bound, upper_bound, timeout, enable_verify)

    @property
    def sat_model(self) -> IncrementalSATModel:
        """Get the SAT model."""
        return self._sat_model

    def solve(self):
        assumptions = list(self._assumptions)

        self._sat_model.solver.clear_interrupt()
        timer = None  # Initialize timer

        if self._timeout is not None:
            def interrupt(s):
                s.interrupt()

            timer = Timer(self._timeout - self._time_used, interrupt, [self._sat_model.solver])
            timer.start()

        try:
            result = self._sat_model.solver.solve_limited(assumptions=assumptions,
                                                          expect_interrupt=True)
            self._time_used = self._sat_model.solver.time_accum()
            return result
        finally:
            if timer:
                timer.cancel()  # Cancel the timer if solve_limited finished early

    def decrease_makespan(self):
        """This method is used to decrease the makespan of the problem.
        It should be called after encode() and solve() methods.
        After calling this method, you will need to call solve() method to solve problem with new makespan."""
        if self._makespan == self._lower_bound:
            raise Exception("Makespan cannot be decreased below the lower bound.")

        last_job = []
        for i in range(self._problem.number_of_activities):
            if self._problem.successors[i] == [self._problem.number_of_activities - 1]:
                last_job.append(i)

        actual_makespan = 0
        for job in last_job:
            if self.solution[job] + self._problem.durations[job] > actual_makespan:
                actual_makespan = self.solution[job] + self._problem.durations[job]

        if self._makespan == actual_makespan:
            self._assumptions.add(
                -self._start[self._problem.number_of_activities - 1, self._makespan])
            self._makespan -= 1
        else:
            for m in range(actual_makespan + 1, self._makespan + 1):
                self._assumptions.add(-self._start[self._problem.number_of_activities - 1, m])
            self._makespan = actual_makespan if actual_makespan > self._lower_bound else self._lower_bound


        self._solution = None

    @property
    def makespan(self) -> int:
        return self._makespan
