from queue import Queue


class PreprocessingFailed(Exception):
    """Exception raised when preprocessing fails."""
    pass


class RCPSPEncoder:
    def __init__(self, problem, makespan, timeout=None, enable_verify=False):
        self.problem = problem
        self.makespan = makespan
        # For each activity: earliest start, earliest close, latest start, and latest close time
        self.ES = [0] * self.problem.njobs
        self.EC = [0] * self.problem.njobs
        self.LS = [self.makespan] * self.problem.njobs
        self.LC = [self.makespan] * self.problem.njobs

        self.assumptions = set()

        self.time_out = timeout
        self.time_used = 0

        self.enable_verify = enable_verify
        self.solution = None

    def _create_variable(self):
        """Create variables for the problem."""
        raise NotImplementedError()

    def _calculate_time_windows(self):
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

    def _preprocessing(self):
        """Preprocess the problem.
        This method should be called before encoding the problem.
        """
        raise NotImplementedError()

    def encode(self):
        """Encode the problem."""
        raise NotImplementedError()

    def solve(self):
        """Solve the problem with the current makespan.
        This method should be called after encode() method."""
        raise NotImplementedError()

    def decrease_makespan(self):
        """Decrease the makespan of the problem.
        This method should be called after encode() and solve() methods.
        After calling this method, you will need to call solve() method to solve problem with new makespan."""
        raise NotImplementedError()

    def get_solution(self):
        """Get the result of the problem where the result is a list of start times for each activity."""
        raise NotImplementedError()

    def verify(self):
        """Verify the solution of the problem."""
        if not self.enable_verify:
            raise Exception(
                "Verification is not enabled. Set enable_verify to True to enable verification.")
        # Get start time
        solution = self.get_solution()

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
