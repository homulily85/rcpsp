import os
import secrets
import shutil
import string
import subprocess
import timeit

from encoder.model.sat_model import MaxSATModel
from encoder.sat.sat_solver import SATSolver
from encoder.model.status import SOLVER_STATUS


class MaxSATSolver(SATSolver):
    def __init__(self, input_file: str,
                 lower_bound: int,
                 upper_bound: int,
                 timeout: int = None,
                 enable_verify: bool = False):
        self.__output_file = None
        self.__return_code = None
        self.__makespan_var = {}
        self._sat_model = MaxSATModel()
        self._register: dict[tuple[int, ...], int] = {}
        super().__init__(input_file, lower_bound, upper_bound, timeout, enable_verify)

    def _create_variable(self):
        super()._create_variable()
        for i in range(self._lower_bound, self._upper_bound + 1):
            self.__makespan_var[i] = self.sat_model.create_new_variable()

    @property
    def sat_model(self) -> MaxSATModel:
        """Get the SAT model."""
        return self._sat_model

    @property
    def makespan(self) -> int:
        """Get the makespan of the solution."""
        if self.__return_code in [
            SOLVER_STATUS.UNSATISFIABLE.value, SOLVER_STATUS.UNKNOWN.value]:
            raise Exception("No solution found")
        return self.solution[-1]

    @staticmethod
    def _generate_random_filename():
        characters = string.ascii_letters + string.digits
        random_str = ''.join(secrets.choice(characters) for _ in range(5))
        return f"{random_str}.wcnf"

    def solve(self):
        # Check if wcnf folder exists
        if os.path.exists('wcnf'):
            shutil.rmtree('wcnf')
        if os.path.exists('out'):
            shutil.rmtree('out')

        os.makedirs('wcnf')
        os.makedirs('out')

        file_name = self._generate_random_filename()
        self.__output_file = 'out/' + file_name + '.out'
        # Export the SAT model to wcnf folder
        self._sat_model.export('wcnf/' + file_name)
        size = os.path.getsize('wcnf/' + file_name)

        if size > 25 * 1024 * 1024:
            self.timeout += 30

        # Solve the problem using the MaxSAT solver
        start = timeit.default_timer()
        if self.timeout is not None:
            command = (
                f"timeout -s SIGTERM {self.timeout}s ./bin/tt-open-wbo-inc-Glucose4_1_static "
                f"wcnf/{file_name} > {self.__output_file}"
            )
        else:
            command = (
                f"./bin/tt-open-wbo-inc-Glucose4_1_static wcnf/{file_name} > {self.__output_file}"
            )

        process = subprocess.Popen(command, shell=True)
        process.wait()

        self.time_used = timeit.default_timer() - start

        last_s_line = ""

        with open(self.__output_file, 'r') as f:
            for line in f:
                if line.startswith('s '):
                    # Keep only the last 's' line
                    last_s_line = line[2:].strip()

        if last_s_line == "UNSATISFIABLE":
            self.__return_code = SOLVER_STATUS.UNSATISFIABLE
        elif last_s_line == "SATISFIABLE":
            self.__return_code = SOLVER_STATUS.SATISFIABLE
        elif last_s_line == "UNKNOWN":
            self.__return_code = SOLVER_STATUS.UNKNOWN
        else:
            self.__return_code = SOLVER_STATUS.OPTIMUM

        return self.__return_code

    @property
    def solution(self) -> list[int]:
        if self.__return_code in [
            SOLVER_STATUS.UNSATISFIABLE.value, SOLVER_STATUS.UNKNOWN.value]:
            raise Exception("No solution found")

        # Parse the solution from the output file
        if self._solution is None:
            self._solution = self._parse_solution()
        return self._solution

    def _parse_solution(self) -> list[int]:
        """
        Parse the solution from the output file and extract start times for each activity.

        Returns:
            list[int]: A list where index i contains the start time of activity i.
        """

        # Read the solution from the output file
        last_v_line = ""
        with open(self.__output_file, 'r') as f:
            for line in f:
                if line.startswith('v '):
                    # Keep only the last 'v' line
                    last_v_line = line[2:].strip()

        # Use the last "v" line as the solution
        solution_string = last_v_line

        # Check if we have a valid solution
        if not solution_string:
            raise Exception("No solution found in output file")

        # Initialize the result list with default values
        result = [-1] * self._problem.number_of_activities

        # Extract the start times from the solution
        for (job, time), var_id in self._start.items():
            if var_id <= len(solution_string) and solution_string[var_id - 1] == '1':
                result[job] = time

        return result

    def _precedence_constraint(self):
        for predecessor in range(1, self._problem.number_of_activities):
            for successor in self._problem.successors[predecessor]:
                # Successor can only start at one time
                self._sat_model.add_clause(
                    [self._get_forward_staircase_register(successor, self._ES[successor],
                                                          self._LS[successor] + 1)])
                # Predecessor can only start at one time
                self._sat_model.add_clause(
                    [self._get_forward_staircase_register(predecessor, self._ES[predecessor],
                                                          self._LS[predecessor] + 1)])

                # Precedence constraint
                for k in range(self._ES[successor], self._LS[successor] + 1):
                    first_half = self._get_forward_staircase_register(successor,
                                                                      self._ES[successor],
                                                                      k + 1)

                    if first_half is None or k - self._problem.durations[predecessor] + 1 >= \
                            self._LS[
                                predecessor] + 1:
                        continue

                    self._sat_model.add_clause(
                        [-first_half,
                         -self._start[predecessor, k - self._problem.durations[predecessor] + 1]])
                else:
                    for k in range(self._LS[successor] - self._problem.durations[predecessor] + 2,
                                   self._LS[predecessor] + 1):
                        self._sat_model.add_clause(
                            [-self._get_forward_staircase_register(successor, self._ES[successor],
                                                                   self._LS[successor] + 1),
                             -self._start[predecessor, k]])

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

    def __hard_constraint(self):
        for t in range(self._lower_bound, self._upper_bound + 1):
            if self._ES[self._problem.number_of_activities - 1] <= t <= self._LS[
                self._problem.number_of_activities - 1]:
                self.sat_model.add_clause(
                    [self.__makespan_var[t],
                     -self._start[self._problem.number_of_activities - 1, t]])

        for t in range(self._lower_bound, self._upper_bound):
            self.sat_model.add_clause([self.__makespan_var[t], -self.__makespan_var[t + 1]])

    def __soft_constraint(self):
        for t in range(self._lower_bound, self._upper_bound + 1):
            self.sat_model.add_soft_clause([-self.__makespan_var[t]], 1)

    def encode(self):
        super().encode()
        self.__hard_constraint()
        self.__soft_constraint()
