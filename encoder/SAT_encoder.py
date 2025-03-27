from queue import Queue
from threading import Timer

from encoder.SAT_model import SATModel
from encoder.problem import Problem


class PreprocessingFailed(Exception):
    """Exception raised when preprocessing fails."""
    pass


class Encoder:
    def __init__(self, problem: Problem, makespan: int, timeout: int = None,
                 enable_verify: bool = False):
        self.time_out = timeout
        self.enable_verify = enable_verify
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

        self.register: dict[tuple[int, ...], int] = {}

        self.assumptions = set()
        self._preprocessing()

        self.time_used = 0
        self.solution = None

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

        for j in range(1, self.problem.njobs):
            for i in self.problem.successors[j]:
                if self.ES[j] + self.problem.durations[j] > self.ES[i]:
                    self.ES[i] = self.ES[j] + self.problem.durations[j]
                    self.EC[i] = self.ES[i] + self.problem.durations[i]

    def encode(self):
        """Encode the problem into SAT clauses."""
        raise NotImplementedError()

    def solve(self):
        """Solve the problem with the current makespan.
        This method should be called after encode() method."""
        assumptions = list(self.assumptions)

        self.sat_model.solver.clear_interrupt()
        timer = None  # Initialize timer

        if self.time_out is not None:
            def interrupt(s):
                s.interrupt()

            timer = Timer(self.time_out, interrupt, [self.sat_model.solver])
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
        for consistency_variable in self.x.keys():
            if self.makespan in consistency_variable:
                self.assumptions.add(-self.x[consistency_variable])
        for start_variable in self.y.keys():
            if start_variable[1] >= self.makespan:
                self.assumptions.add(-self.y[start_variable])
        self.makespan -= 1
        self.solution = None

    def get_result(self) -> list[int]:
        """Get the result of the problem where the result is a list of start times for each activity."""
        if self.solution is not None:
            return self.solution

        if self.sat_model.solver.get_model() is None:
            raise Exception(
                f"This problem is unsatisfiable with the current makespan {self.makespan}")
        sol = []
        for i in range(self.problem.njobs):
            for s in range(self.ES[i], self.LS[i] + 1):
                if self.sat_model.solver.get_model()[self.y[(i, s)] - 1] > 0:
                    sol.append(s)
                    break
        return sol

    def verify(self):
        """Verify the solution of the problem."""
        if not self.enable_verify:
            return
        # Get start time
        solution = self.get_result()

        # Check precedence constraint
        for job in range(self.problem.njobs):
            for predecessor in self.problem.predecessors[job]:
                if solution[job] < solution[predecessor] + self.problem.durations[predecessor]:
                    print(
                        f"Failed when checking precedence constraint for {predecessor} -> {job}"
                        f" while checking {self.problem.name}")
                    exit(-1)

        # Checking resource constraint
        for t in range(solution[-1] + 1):
            for r in range(self.problem.nresources):
                total_consume = 0
                for j in range(self.problem.njobs):
                    if solution[j] <= t <= solution[j] + self.problem.durations[j] - 1:
                        total_consume += self.problem.requests[j][r]
                if total_consume > self.problem.capacities[r]:
                    print(f"Failed when check resource constraint for resource {r} at t = {t}"
                          f" while checking {self.problem.name}")
                    exit(-1)
