from encoder.model.weighted_at_most_k_model import WeightedAtMostKModel
from encoder.incremental_sat.SAT_encoder import SATSolver


class StaircaseSATEncoder(SATSolver):
    def __init__(self, problem, makespan: int, timeout: int = None, enable_verify: bool = False):
        super().__init__(problem, makespan, timeout, enable_verify)
        self.register: dict[tuple[int, ...], int] = {}

    def encode(self):
        self._resource_constraint()
        self._start_time_for_job_0()
        self._precedence_constraint()
        self._consistency_constraint()
        self._redundant_constraint()

    def _precedence_constraint(self):
        for predecessor in range(1, self.problem.number_of_activities):
            for successor in self.problem.successors[predecessor]:
                # Successor can only start at one time
                self.sat_model.add_clause(
                    [self._get_forward_staircase_register(successor, self.ES[successor],
                                                          self.LS[successor] + 1)])
                # Predecessor can only start at one time
                self.sat_model.add_clause(
                    [self._get_backward_staircase_register(predecessor, self.ES[predecessor],
                                                           self.LS[predecessor] + 1)])

                # Precedence constraint
                for k in range(self.ES[successor], self.LS[successor] + 1):
                    first_half = self._get_forward_staircase_register(successor,
                                                                      self.ES[successor],
                                                                      k + 1)
                    second_half = self._get_backward_staircase_register(predecessor,
                                                                        k - self.problem.durations[
                                                                            predecessor] + 1,
                                                                        self.LS[predecessor] + 1)
                    if first_half is None or second_half is None:
                        continue

                    self.sat_model.add_clause([-first_half, -second_half])

    def _get_forward_staircase_register(self, job: int, start: int, end: int) -> int | None:
        """Get forward staircase register for a job for a range of time [start, end)"""
        # Check if current job with provided start and end time is already in the register
        if start >= end:
            return None
        temp = tuple(self.start[(job, s)] for s in range(start, end))
        if temp in self.register:
            return self.register[temp]

        # Store the start time of the job
        accumulative = []
        for s in range(start, end):
            accumulative.append(self.start[(job, s)])
            current_tuple = tuple(accumulative)
            # If the current tuple is not in the register, create a new variable
            if current_tuple not in self.register:
                # Create a new variable for the current tuple
                self.register[current_tuple] = self.sat_model.create_new_variable()

                # Create constraint for staircase

                # If current tuple is true then the register associated with it must be true
                self.sat_model.add_clause(
                    [-self.start[(job, s)], self.register[current_tuple]])

                if s == start:
                    self.sat_model.add_clause(
                        [self.start[(job, s)], -self.register[current_tuple]])
                else:
                    # Get the previous tuple
                    previous_tuple = tuple(accumulative[:-1])
                    # If previous tuple is true then the current register must be true
                    self.sat_model.add_clause(
                        [-self.register[previous_tuple], self.register[current_tuple]])

                    # Both previous tuple and current variable is false then current tuple must be false
                    self.sat_model.add_clause(
                        [self.register[previous_tuple], self.start[job, s],
                         -self.register[current_tuple]])

                    # Previous tuple and current variable must not be true at the same time
                    self.sat_model.add_clause(
                        [-self.register[previous_tuple], -self.start[(job, s)]])

        return self.register[temp]

    def _get_backward_staircase_register(self, job: int, start: int, end: int) -> int | None:
        """Get backward staircase register for a job for a range of time [start, end)"""
        # Check if current job with provided start and end time is already in the register
        if start >= end:
            return None
        temp = tuple(self.start[(job, s)] for s in range(end - 1, start - 1, -1))
        if temp in self.register:
            return self.register[temp]

        accumulative = []
        for s in range(end - 1, start - 1, -1):
            accumulative.append(self.start[(job, s)])
            current_tuple = tuple(accumulative)
            # If the current tuple is not in the register, create a new variable
            if current_tuple not in self.register:
                # Create a new variable for the current tuple
                self.register[current_tuple] = self.sat_model.create_new_variable()

                # Create constraint for staircase

                # If current tuple is true then the register associated with it must be true
                self.sat_model.add_clause(
                    [-self.start[(job, s)], self.register[current_tuple]])

                if s == end - 1:
                    self.sat_model.add_clause(
                        [self.start[(job, s)], -self.register[current_tuple]])

                else:
                    # Get the previous tuple
                    previous_tuple = tuple(accumulative[:-1])
                    # If previous tuple is true then the current register must be true
                    self.sat_model.add_clause(
                        [-self.register[previous_tuple], self.register[current_tuple]])

                    # Both previous tuple and current variable is false then current tuple must be false
                    self.sat_model.add_clause(
                        [self.register[previous_tuple], self.start[job, s],
                         -self.register[current_tuple]])

                    # Previous tuple and current variable must not be true at the same time
                    self.sat_model.add_clause(
                        [-self.register[previous_tuple], -self.start[(job, s)]])

        return self.register[temp]

    def _resource_constraint(self):
        for t in range(self.makespan):
            for r in range(self.problem.number_of_resources):
                pb_constraint = WeightedAtMostKModel(self.sat_model, self.problem.capacities[r])
                for i in range(self.problem.number_of_activities):
                    if t in range(self.ES[i], self.LC[i] + 1):
                        pb_constraint.add_term(self.run[(i, t)], self.problem.requests[i][r])
                pb_constraint.encode()

    def _redundant_constraint(self):
        for i in range(self.problem.number_of_activities):
            for c in range(self.EC[i], self.LC[i]):
                self.sat_model.add_clause(
                    [-self.run[(i, c)], self.run[(i, c + 1)],
                     self.start[(i, c - self.problem.durations[i] + 1)]])

    def _consistency_constraint(self):
        for i in range(self.problem.number_of_activities):
            for s in range(self.ES[i], self.LS[i] + 1):  # s in STW(i)
                for t in range(s, s + self.problem.durations[i]):
                    self.sat_model.add_clause(
                        [-self.start[(i, s)], self.run[(i, t)]])
                    self.sat_model.number_of_consistency_clauses += 1

    def _start_time_for_job_0(self):
        self.sat_model.add_clause([self.start[(0, 0)]])
        # temp = []
        # for job in range(self.problem.number_of_activities):
        #     if self.problem.predecessors[job] == [0]:
        #         temp.append(job)
        # self.sat_model.add_clause([self.start[job, 0] for job in temp])
