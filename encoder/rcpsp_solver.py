from encoder.model.problem import Problem


class PreprocessingFailed(Exception):
    """Exception raised when preprocessing fails."""
    pass


class RCPSPSolver:
    """
    Base class for solving the Resource-Constrained Project Scheduling Problem (RCPSP).
    """

    def __init__(self, input_file: str,
                 timeout: int = None,
                 enable_verify: bool = False):
        """
        Initialize the RCPSP solver with the input file, _timeout, and verification flag.
        :param input_file: Path to the input file.
        :type input_file: str
        :param timeout: Timeout for the solver in seconds.
        :type timeout: int|None
        :param enable_verify: Flag to enable verification of the solution.
        :type enable_verify: bool
        :raises ValueError: If the timeout is negative.
        """
        self._input_file = input_file
        if timeout is not None and timeout < 0:
            raise ValueError("Timeout must be a non-negative integer.")
        self._timeout = timeout
        self._time_used = 0
        self._enable_verify = enable_verify
        self._solution = None

    @property
    def time_used(self) -> float:
        """
        Get the time used by the solver.
        :return: The time used by the solver in seconds.
        :rtype: float
        """
        return self._time_used

    @time_used.setter
    def time_used(self, time_used: float):
        """
        Set the time used by the solver.
        :param time_used: The time used by the solver in seconds.
        :type time_used: float
        """
        self._time_used = time_used

    @property
    def input_file(self) -> str:
        """
        Get the input file.
        :return: The input file path.
        :rtype: str
        """
        return self._input_file

    @property
    def timeout(self) -> int:
        """
        Get the timeout.
        :return: The timeout in seconds.
        :rtype: int
        """
        return self._timeout

    @timeout.setter
    def timeout(self, timeout: int):
        """
        Set the timeout.
        :param timeout: The timeout in seconds.
        :type timeout: int
        """
        if timeout < 0:
            raise ValueError("Timeout must be a non-negative integer.")
        self._timeout = timeout

    @property
    def makespan(self) -> int:
        """
        Get the makespan of the solution.
        :return: The makespan of the solution.
        :rtype: int
        """
        raise NotImplementedError()

    @property
    def solution(self) -> list[int]:
        """
        Get the result of the problem where the result is a list of start times for each activity.
        :return: The solution of the problem.
        :rtype: list[int]
        """
        raise NotImplementedError()

    def solve(self):
        """
        Solve the problem.
        """
        raise NotImplementedError()

    def verify(self):
        """
        Verify the solution of the problem.
        """
        if not self._enable_verify:
            raise Exception("Verification is not enabled.")

        # Get start time
        solution = self.solution
        problem = Problem(self._input_file)

        # Check precedence constraint
        for job in range(problem.number_of_activities):
            for predecessor in problem.predecessors[job]:
                if solution[job] < solution[predecessor] + problem.durations[predecessor]:
                    print(
                        f"Failed when checking precedence constraint for {predecessor} -> {job}"
                        f" while checking {problem.name}")
                    exit(-1)

        # Checking resource constraint
        for t in range(solution[-1] + 1):
            for r in range(problem.number_of_resources):
                total_consume = 0
                for j in range(problem.number_of_activities):
                    if solution[j] <= t <= solution[j] + problem.durations[j] - 1:
                        total_consume += problem.requests[j][r]
                if total_consume > problem.capacities[r]:
                    print(f"Failed when check resource constraint for resource {r} at t = {t}"
                          f" while checking {problem.name}")
                    exit(-1)
