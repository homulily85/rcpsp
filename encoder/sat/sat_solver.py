from queue import Queue

from encoder.model.problem import Problem
from encoder.model.weighted_at_most_k_model import WeightedAtMostKModel
from encoder.rcpsp_solver import RCPSPSolver, PreprocessingFailed


class SATSolver(RCPSPSolver):
    """
    Base class for solving the Resource-Constrained Project Scheduling Problem (RCPSP) using SAT encoding.
    """

    def __init__(self, input_file: str,
                 lower_bound: int,
                 upper_bound: int,
                 timeout: int = None,
                 enable_verify: bool = False):
        super().__init__(input_file, timeout, enable_verify)

        self._problem = Problem(input_file)

        self._lower_bound = lower_bound
        self._upper_bound = upper_bound

        # For each activity: earliest start, earliest close, latest start, and latest close time
        self._ES = [0] * self._problem.number_of_activities
        self._EC = [0] * self._problem.number_of_activities
        self._LS = [self._upper_bound] * self._problem.number_of_activities
        self._LC = [self._upper_bound] * self._problem.number_of_activities
        # Variable start_(i,t): boolean representing whether activity i starts at time t in STW(i)
        self._start: dict[tuple[int, int], int] = {}
        # Variable run_(i,t): boolean representing whether activity i is running at time t in RTW(i)
        self._run: dict[tuple[int, int], int] = {}
        self._preprocessing()

    @property
    def sat_model(self):
        raise NotImplementedError()

    def _calculate_time_windows(self):
        """Calculate the earliest and latest start and close times for each activity."""
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

    def _preprocessing(self):
        """Preprocess the problem to calculate time windows and initialize variables."""
        self._calculate_time_windows()
        self._create_variable()

    def _create_variable(self):
        """Create variables for the SAT model."""
        for i in range(self._problem.number_of_activities):
            for t in range(self._ES[i],
                           self._LS[i] + 1):  # t in STW(i) (start time window of activity i)
                self._start[(i, t)] = self.sat_model.create_new_variable()

        for i in range(self._problem.number_of_activities):
            for t in range(self._ES[i],
                           self._LC[i] + 1):  # t in RTW(i) (run time window of activity i)
                self._run[(i, t)] = self.sat_model.create_new_variable()

    def encode(self):
        """Encode the problem into SAT clauses."""
        self._start_time_for_job_0()
        self._precedence_constraint()
        self._consistency_constraint()
        self._redundant_constraint()
        self._resource_constraint()

    @property
    def solution(self) -> list[int]:
        if self.sat_model.solver.get_model() is None:
            raise Exception(
                f"This problem is unsatisfiable.")

        if self._solution is not None:
            return self._solution

        sol = []
        for i in range(self._problem.number_of_activities):
            start_time_found = False
            for s in range(self._ES[i], self._LS[i] + 1):
                if self.sat_model.solver.get_model()[self._start[(i, s)] - 1] > 0:
                    sol.append(s)
                    start_time_found = True
                    break
            if not start_time_found:
                raise Exception(
                    f"Start time for activity {i} not found in the solution."
                )
        self._solution = sol
        return sol

    def _resource_constraint(self):
        for t in range(self._upper_bound):
            for r in range(self._problem.number_of_resources):
                pb_constraint = WeightedAtMostKModel(self.sat_model, self._problem.capacities[r])
                for i in range(self._problem.number_of_activities):
                    if t in range(self._ES[i], self._LC[i] + 1):
                        pb_constraint.add_term(self._run[(i, t)], self._problem.requests[i][r])
                pb_constraint.encode()

    def _redundant_constraint(self):
        for i in range(self._problem.number_of_activities):
            for c in range(self._EC[i], self._LC[i]):
                self.sat_model.add_clause(
                    [-self._run[(i, c)], self._run[(i, c + 1)],
                     self._start[(i, c - self._problem.durations[i] + 1)]])

    def _consistency_constraint(self):
        for i in range(self._problem.number_of_activities):
            for s in range(self._ES[i], self._LS[i] + 1):  # s in STW(i)
                for t in range(s, s + self._problem.durations[i]):
                    self.sat_model.add_clause(
                        [-self._start[(i, s)], self._run[(i, t)]])
                    self.sat_model.number_of_consistency_clauses += 1

    def _start_time_for_job_0(self):
        self.sat_model.add_clause([self._start[(0, 0)]])
        # temp = []
        # for job in range(self.problem.number_of_activities):
        #     if self.problem.predecessors[job] == [0]:
        #         temp.append(job)
        # self.sat_model.add_clause([self.start[job, 0] for job in temp])

    def _precedence_constraint(self):
        raise NotImplementedError()
