from enum import auto
from queue import Queue

from pip._internal.utils.misc import enum

from src.model import Problem, SATSolver
from src.utils import SimpleLogger


class PreprocessingFailed(Exception):
    """Exception raised when preprocessing fails."""
    pass


class RcpspIncrementalSatSolver:
    """
    Base class for solving the Resource-Constrained Project Scheduling Problem (RCPSP) using incremental SAT method.
    """

    def __init__(self, input_file: str,
                 lower_bound: int,
                 upper_bound: int,
                 timeout: int = None,
                 enable_verify: bool = False):
        """
        Initialize the RCPSP solver with the input file, timeout, and verification flag.
        :param input_file: Path to the input file.
        :type input_file: str
        :param lower_bound: Lower bound for the makespan.
        :type lower_bound: int
        :param upper_bound: Upper bound for the makespan.
        :type upper_bound: int
        :param timeout: Timeout for the solver in seconds.
        :type timeout: int|None
        :param enable_verify: Flag to enable verification of the solution.
        :type enable_verify: bool
        :raises ValueError: If the timeout is negative.
        """
        self._problem = Problem(input_file)
        self._enable_verify = enable_verify

        if lower_bound < 0 or upper_bound < 0:
            raise ValueError("Lower bound and upper bound must be a non-negative integer.")
        if upper_bound < lower_bound:
            raise ValueError("Upper bound must be greater than or equal to lower bound.")
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound
        self._makespan = self._upper_bound

        # For each activity: earliest start, earliest close, latest start, and latest close time
        self._ES = [0] * self._problem.number_of_activities
        self._EC = [0] * self._problem.number_of_activities
        self._LS = [self._upper_bound] * self._problem.number_of_activities
        self._LC = [self._upper_bound] * self._problem.number_of_activities

        # Variable start_(i,t): boolean representing whether activity i starts at time t in STW(i)
        self._start: dict[tuple[int, int], int] = {}
        # Variable run_(i,t): boolean representing whether activity i is running at time t in RTW(i)
        self._run: dict[tuple[int, int], int] = {}

        self._solver = SATSolver(timeout=timeout)
        self._solution = None

        self._preprocessing()

    def _preprocessing(self):
        """Preprocess the problem to calculate time windows and initialize variables."""
        SimpleLogger().log("Preprocessing the problem...")
        self._calculate_time_windows()
        self._create_variable()
        SimpleLogger().log("Preprocessing completed.")

    def _create_variable(self):
        """Create variables for the SAT model."""
        SimpleLogger().log("Creating variables for the SAT model...")
        for i in range(self._problem.number_of_activities):
            for t in range(self._ES[i],
                           self._LS[i] + 1):  # t in STW(i) (start time window of activity i)
                self._start[(i, t)] = self._solver.create_new_variable()

        for i in range(self._problem.number_of_activities):
            for t in range(self._ES[i],
                           self._LC[i] + 1):  # t in RTW(i) (run time window of activity i)
                self._run[(i, t)] = self._solver.create_new_variable()

        SimpleLogger().log("Variables created successfully.")

    def _calculate_time_windows(self):
        """Calculate the earliest and latest start and close times for each activity."""
        SimpleLogger().log("Calculating time windows for each activity...")
        # Calculate ES and EC
        mark = [False] * self._problem.number_of_activities
        queue = Queue()
        queue.put(0)

        while not queue.empty():
            curr_job = queue.get()
            mark[curr_job] = True

            self._ES[curr_job] = 0 if len(self._problem.predecessors[curr_job]) == 0 else max(
                self._EC[p] for p in self._problem.predecessors[curr_job])

            self._EC[curr_job] = self._ES[curr_job] + self._problem.durations[curr_job]
            if self._EC[curr_job] > self._upper_bound:
                raise PreprocessingFailed

            for successor in self._problem.successors[curr_job]:
                if not mark[successor]:
                    queue.put(successor)

        # Calculate LC and LS
        mark = [False] * self._problem.number_of_activities
        queue = Queue()
        queue.put(self._problem.number_of_activities - 1)

        while not queue.empty():
            curr_job = queue.get()
            mark[curr_job] = True

            self._LC[curr_job] = self._upper_bound if len(
                self._problem.successors[curr_job]) == 0 else min(
                self._LS[s] for s in self._problem.successors[curr_job])

            self._LS[curr_job] = self._LC[curr_job] - self._problem.durations[curr_job]
            if self._LS[curr_job] < 0:
                raise PreprocessingFailed

            for predecessor in self._problem.predecessors[curr_job]:
                if not mark[predecessor]:
                    queue.put(predecessor)

        for j in range(1, self._problem.number_of_activities):
            for i in self._problem.successors[j]:
                if self._ES[j] + self._problem.durations[j] > self._ES[i]:
                    self._ES[i] = self._ES[j] + self._problem.durations[j]
                    self._EC[i] = self._ES[i] + self._problem.durations[i]

        SimpleLogger().log("Time windows calculated successfully.")

    @property
    def makespan(self) -> int:
        """
        Get the makespan of the solution.
        :return: The makespan of the solution.
        :rtype: int
        """
        return self._makespan

    @property
    def solution(self) -> list[int]:
        """
        Get the result of the problem where the result is a list of start times for each activity.
        :return: The solution of the problem.
        :rtype: list[int]
        """
        if self._solver.get_model() is None:
            raise Exception(f"This problem is unsatisfiable.")

        SimpleLogger().log("Retrieving the solution...")
        if self._solution is not None:
            return self._solution

        start_times = [-1 for _ in range(self._problem.number_of_activities)]

        def get_start_time(activity: int) -> int:
            for s in range(self._ES[activity], self._LS[activity] + 1):
                if self._solver.get_model()[self._start[(activity, s)] - 1] > 0:
                    return s
            raise Exception(f"Start time for activity {activity} not found in the solution.")

        from concurrent.futures import ThreadPoolExecutor, as_completed

        # Set your desired max number of threads
        import os
        MAX_THREADS = os.cpu_count()

        # Function wrapper if needed
        def get_start_time_wrapper(i):
            return i, get_start_time(i)

        with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
            # Submit all tasks
            futures = {executor.submit(get_start_time_wrapper, i): i for i in
                       range(self._problem.number_of_activities)}

            # Collect results as they complete
            for future in as_completed(futures):
                i, start_time = future.result()
                start_times[i] = start_time

        self._solution = start_times
        SimpleLogger().log("Solution retrieved successfully.")
        return start_times

    def solve(self):
        """
        Solve the problem.
        """
        SimpleLogger().log("Solving the problem...")
        t = self._solver.solve()
        SimpleLogger().log("Finished solving the problem.")
        return t

    def verify(self):
        """
        Verify the solution of the problem.
        """
        # Get start time
        if not self._enable_verify:
            SimpleLogger().log("Verification is disabled.")
            return

        SimpleLogger().log("Verifying the solution...")
        solution = self.solution

        # Check precedence constraint
        for job in range(self._problem.number_of_activities):
            for predecessor in self._problem.predecessors[job]:
                if solution[job] < solution[predecessor] + self._problem.durations[predecessor]:
                    SimpleLogger().log(
                        f"Failed when checking precedence constraint for {predecessor} -> {job}"
                        f" while checking {self._problem.name}")
                    exit(-1)

        # Checking resource constraint
        for t in range(solution[-1] + 1):
            for r in range(self._problem.number_of_resources):
                total_consume = 0
                for j in range(self._problem.number_of_activities):
                    if solution[j] <= t <= solution[j] + self._problem.durations[j] - 1:
                        total_consume += self._problem.requests[j][r]
                if total_consume > self._problem.capacities[r]:
                    SimpleLogger().log(
                        f"Failed when check resource constraint for resource {r} at t = {t}"
                        f" while checking {self._problem.name}")
                    exit(-1)

        SimpleLogger().log("Solution verified successfully.")

    def decrease_makespan(self):
        """This method is used to decrease the makespan of the problem.
        It should be called after encode() and solve() methods.
        After calling this method, you will need to call solve() method to solve problem with new makespan."""
        if self._makespan == self._lower_bound:
            raise Exception("Makespan cannot be decreased below the lower bound.")

        self._solver.add_assumption(
            -self._start[self._problem.number_of_activities - 1, self._makespan])

        self._makespan -= 1
        self._solution = None


class StaircaseMode(enum):
    """
    Enum for staircase encoding modes.
    """
    FORWARD = auto()
    BACKWARD = auto()
    BOTH = auto()


class RcpspStaircaseIncrementalSatSolver(RcpspIncrementalSatSolver):
    """
    Class for solving the Resource-Constrained Project Scheduling Problem (RCPSP) using incremental SAT method with staircase encoding.
    """

    def __init__(self, input_file: str,
                 lower_bound: int,
                 upper_bound: int,
                 timeout: int = None,
                 enable_verify: bool = False,
                 mode: StaircaseMode = StaircaseMode.FORWARD):
        """
        Initialize the RCPSP solver with the input file, timeout, and verification flag.
        :param input_file: Path to the input file.
        :type input_file: str
        :param lower_bound: Lower bound for the makespan.
        :type lower_bound: int
        :param upper_bound: Upper bound for the makespan.
        :type upper_bound: int
        :param timeout: Timeout for the solver in seconds.
        :type timeout: int|None
        :param enable_verify: Flag to enable verification of the solution.
        :type enable_verify: bool
        """
        super().__init__(input_file, lower_bound, upper_bound, timeout, enable_verify)
        self.__mode = mode

    def _encode(self):
        """Encode the problem into SAT clauses."""
        self._start_time_for_job_0()
        self._precedence_constraint()
        self._consistency_constraint()
        self._redundant_constraint()
        self._resource_constraint()

    def _start_time_for_job_0(self):
        self._solver.add_clause([self._start[(0, 0)]])
        # temp = []
        # for job in range(self._problem.number_of_activities):
        #     if self._problem.predecessors[job] == [0]:
        #         temp.append(job)
        # self.sat_model.add_clause([self._start[job, 0] for job in temp])

    def _consistency_constraint(self):
        for i in range(self._problem.number_of_activities):
            for s in range(self._ES[i], self._LS[i] + 1):  # s in STW(i)
                for t in range(s, s + self._problem.durations[i]):
                    self._solver.add_clause(
                        [-self._start[(i, s)], self._run[(i, t)]])

    def _redundant_constraint(self):
        for i in range(self._problem.number_of_activities):
            for c in range(self._EC[i], self._LC[i]):
                self._solver.add_clause(
                    [-self._run[(i, c)], self._run[(i, c + 1)],
                     self._start[(i, c - self._problem.durations[i] + 1)]])

    def _resource_constraint(self):
        for t in range(self._upper_bound + 1):
            for r in range(self._problem.number_of_resources):
                for i in range(self._problem.number_of_activities):
                    self._solver.add_at_most_k([self._run[(i, t)] for t in range(self._ES[i], self._LC[i] + 1)],
                                               [self._problem.requests[i][r] for t in range(self._ES[i], self._LC[i] + 1)],
                                               self._problem.capacities[r])