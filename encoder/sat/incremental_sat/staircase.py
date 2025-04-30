from encoder.sat.incremental_sat.incremental_sat_solver import IncrementalSATSolver


class StaircaseMethod(IncrementalSATSolver):
    """
    Class for encoding the Resource-Constrained Project Scheduling Problem (RCPSP) using the staircase method.
    """

    def __init__(self, input_file: str,
                 lower_bound: int,
                 upper_bound: int,
                 timeout: int = None,
                 enable_verify: bool = False):
        super().__init__(input_file, lower_bound, upper_bound, timeout, enable_verify)
        self._register: dict[tuple[int, ...], int] = {}

    def _precedence_constraint(self):
        for predecessor in range(1, self._problem.number_of_activities):
            for successor in self._problem.successors[predecessor]:
                # Successor can only start at one time
                self._sat_model.add_clause(
                    [self._get_forward_staircase_register(successor, self._ES[successor],
                                                          self._LS[successor] + 1)])
                # Predecessor can only start at one time
                self._sat_model.add_clause(
                    [self._get_backward_staircase_register(predecessor, self._ES[predecessor],
                                                           self._LS[predecessor] + 1)])

                # Precedence constraint
                for k in range(self._ES[successor], self._LS[successor] + 1):
                    first_half = self._get_forward_staircase_register(successor,
                                                                      self._ES[successor],
                                                                      k + 1)
                    second_half = self._get_backward_staircase_register(predecessor,
                                                                        k - self._problem.durations[
                                                                            predecessor] + 1,
                                                                        self._LS[predecessor] + 1)
                    if first_half is None or second_half is None:
                        continue

                    self._sat_model.add_clause([-first_half, -second_half])

    def _get_forward_staircase_register(self, job: int, start: int, end: int) -> int | None:
        """Get forward staircase register for a job for a range of time [start, end)"""
        # Check if current job with provided start and end time is already in the register
        if start >= end:
            return None
        temp = tuple(self._start[(job, s)] for s in range(start, end))
        if temp in self._register:
            return self._register[temp]

        # Store the start time of the job
        accumulative = []
        for s in range(start, end):
            accumulative.append(self._start[(job, s)])
            current_tuple = tuple(accumulative)
            # If the current tuple is not in the register, create a new variable
            if current_tuple not in self._register:
                # Create a new variable for the current tuple
                self._register[current_tuple] = self._sat_model.create_new_variable()

                # Create constraint for staircase

                # If current tuple is true then the register associated with it must be true
                self._sat_model.add_clause(
                    [-self._start[(job, s)], self._register[current_tuple]])

                if s == start:
                    self._sat_model.add_clause(
                        [self._start[(job, s)], -self._register[current_tuple]])
                else:
                    # Get the previous tuple
                    previous_tuple = tuple(accumulative[:-1])
                    # If previous tuple is true then the current register must be true
                    self._sat_model.add_clause(
                        [-self._register[previous_tuple], self._register[current_tuple]])

                    # Both previous tuple and current variable is false then current tuple must be false
                    self._sat_model.add_clause(
                        [self._register[previous_tuple], self._start[job, s],
                         -self._register[current_tuple]])

                    # Previous tuple and current variable must not be true at the same time
                    self._sat_model.add_clause(
                        [-self._register[previous_tuple], -self._start[(job, s)]])

        return self._register[temp]

    def _get_backward_staircase_register(self, job: int, start: int, end: int) -> int | None:
        """Get backward staircase register for a job for a range of time [start, end)"""
        # Check if current job with provided start and end time is already in the register
        if start >= end:
            return None
        temp = tuple(self._start[(job, s)] for s in range(end - 1, start - 1, -1))
        if temp in self._register:
            return self._register[temp]

        accumulative = []
        for s in range(end - 1, start - 1, -1):
            accumulative.append(self._start[(job, s)])
            current_tuple = tuple(accumulative)
            # If the current tuple is not in the register, create a new variable
            if current_tuple not in self._register:
                # Create a new variable for the current tuple
                self._register[current_tuple] = self._sat_model.create_new_variable()

                # Create constraint for staircase

                # If current tuple is true then the register associated with it must be true
                self._sat_model.add_clause(
                    [-self._start[(job, s)], self._register[current_tuple]])

                if s == end - 1:
                    self._sat_model.add_clause(
                        [self._start[(job, s)], -self._register[current_tuple]])

                else:
                    # Get the previous tuple
                    previous_tuple = tuple(accumulative[:-1])
                    # If previous tuple is true then the current register must be true
                    self._sat_model.add_clause(
                        [-self._register[previous_tuple], self._register[current_tuple]])

                    # Both previous tuple and current variable is false then current tuple must be false
                    self._sat_model.add_clause(
                        [self._register[previous_tuple], self._start[job, s],
                         -self._register[current_tuple]])

                    # Previous tuple and current variable must not be true at the same time
                    self._sat_model.add_clause(
                        [-self._register[previous_tuple], -self._start[(job, s)]])

        return self._register[temp]
