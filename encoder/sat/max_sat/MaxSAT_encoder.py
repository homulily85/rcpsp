import os
import secrets
import string
import subprocess
import timeit
from enum import Enum, auto

from encoder.RCPSP_Encoder import RCPSPEncoder
from encoder.problem import Problem
from encoder.sat.max_sat.PB_constraint import PBConstraint
from encoder.sat.max_sat.max_sat_model import MaxSATModel


class SOLVER_STATUS(Enum):
    OPTIMUM = auto()
    UNSATISFIABLE = auto()
    SATISFIABLE = auto()
    UNKNOWN = auto()


class MaxSATEncoder(RCPSPEncoder):
    def __init__(self, problem: Problem, upper_bound: int, lower_bound: int, timeout: int = None,
                 enable_verify: bool = False):
        """Initialize the encoder with the problem, makespan, timeout, and verification flag."""
        super().__init__(problem, upper_bound, timeout, enable_verify)
        self.sat_model = MaxSATModel()
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.return_code = None
        self.output_file = None
        # Variable start_(i,t): boolean representing whether activity i starts at time t in STW(i)
        self.start: dict[tuple[int, int], int] = {}
        # Variable run_(i,t): boolean representing whether activity i is running at time t in RTW(i)
        self.run: dict[tuple[int, int], int] = {}
        self.makespan_var = {}

        self.register: dict[tuple[int, ...], int] = {}
        self._preprocessing()

    def _preprocessing(self):
        self._calculate_time_windows()
        self._create_variable()

    def _create_variable(self):
        for i in range(self.problem.njobs):
            for t in range(self.ES[i],
                           self.LS[i] + 1):  # t in STW(i) (start time window of activity i)
                self.sat_model.number_of_variable += 1
                self.start[(i, t)] = self.sat_model.number_of_variable

        for i in range(self.problem.njobs):
            for t in range(self.ES[i],
                           self.LC[i] + 1):  # t in RTW(i) (run time window of activity i)
                self.sat_model.number_of_variable += 1
                self.run[(i, t)] = self.sat_model.number_of_variable

        for i in range(self.lower_bound, self.upper_bound + 1):
            self.makespan_var[i] = self.sat_model.get_new_var()

    @staticmethod
    def _generate_random_filename():
        characters = string.ascii_letters + string.digits
        random_str = ''.join(secrets.choice(characters) for _ in range(5))
        return f"{random_str}.wcnf"

    def solve(self):
        # Check if wcnf folder exists
        if not os.path.exists('wcnf'):
            os.makedirs('wcnf')

        if not os.path.exists('out'):
            os.makedirs('out')

        file_name = self._generate_random_filename()
        self.output_file = 'out/' + file_name + '.out'
        # Export the SAT model to wcnf folder
        self.sat_model.export('wcnf/' + file_name)
        # Solve the problem using the MaxSAT solver
        start = timeit.default_timer()
        if self.time_out is not None:
            command = (
                f"timeout -s SIGTERM {self.time_out}s ./tt-open-wbo-inc-Glucose4_1_static "
                f"wcnf/{file_name} > {self.output_file}"
            )
        else:
            command = (
                f"./tt-open-wbo-inc-Glucose4_1_static wcnf/{file_name} > {self.output_file}"
            )

        process = subprocess.Popen(command, shell=True)
        return_code = process.wait()

        self.time_used = timeit.default_timer() - start

        match return_code:
            case 30:
                self.return_code = SOLVER_STATUS.OPTIMUM.value
            case 20:
                self.return_code = SOLVER_STATUS.SATISFIABLE.value
            case 10:
                self.return_code = SOLVER_STATUS.UNSATISFIABLE.value
            case _:
                self.return_code = SOLVER_STATUS.UNKNOWN

        return self.return_code

    def get_solution(self) -> list[int]:
        if self.return_code in [
            SOLVER_STATUS.UNSATISFIABLE.value, SOLVER_STATUS.UNKNOWN.value]:
            raise Exception("No solution found")

        # Parse the solution from the output file
        solution = self.parse_solution()
        return solution

    def parse_solution(self) -> list[int]:
        """
        Parse the solution from the output file and extract start times for each activity.
        
        Returns:
            list[int]: A list where index i contains the start time of activity i.
        """

        # Read the solution from the output file
        last_v_line = ""
        with open(self.output_file, 'r') as f:
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
        result = [-1] * self.problem.njobs

        # Extract the start times from the solution
        for (job, time), var_id in self.start.items():
            if var_id <= len(solution_string) and solution_string[var_id - 1] == '1':
                result[job] = time

        self.makespan = max(result)

        return result

    def encode(self):
        self._resource_constraint()
        self._start_time_for_job_0()
        self._precedence_constraint()
        self._consistency_constraint()
        self._redundant_constraint()
        self._soft_constraint()
        self._hard_constraint()

    def _soft_constraint(self):
        for t in range(self.lower_bound, self.upper_bound + 1):
            self.sat_model.add_soft_clause([-self.makespan_var[t]], self.upper_bound - t + 1)

    def _get_last_execute_job(self):
        tmp = []
        for job in range(self.problem.njobs):
            if self.problem.successors[job] == [self.problem.njobs - 1]:
                tmp.append(job)
        return tmp

    def _hard_constraint(self):
        for t in range(self.lower_bound, self.upper_bound + 1):
            if self.ES[self.problem.njobs - 1] <= t <= self.LS[self.problem.njobs - 1]:
                self.sat_model.add_hard_clause(
                    [self.makespan_var[t], -self.start[self.problem.njobs - 1, t]])

        for t in range(self.lower_bound, self.upper_bound):
            self.sat_model.add_hard_clause([self.makespan_var[t], -self.makespan_var[t + 1]])

        last_job = self._get_last_execute_job()
        for t in range(self.lower_bound, self.upper_bound + 1):
            for job in last_job:
                if self.ES[job] <= t <= self.LS[job]:
                    self.sat_model.add_hard_clause([self.makespan_var[t], -self.run[(job, t)]])
                    # self.sat_model.add_hard_clause([-self.makespan_var[t], self.run[(job, t)]])

    def _precedence_constraint(self):
        for predecessor in range(1, self.problem.njobs):
            for successor in self.problem.successors[predecessor]:
                # Successor can only start at one time
                self.sat_model.add_hard_clause(
                    [self._get_forward_staircase_register(successor, self.ES[successor],
                                                          self.LS[successor] + 1)])
                # Predecessor can only start at one time
                self.sat_model.add_hard_clause(
                    [self._get_forward_staircase_register(predecessor, self.ES[predecessor],
                                                          self.LS[predecessor] + 1)])

                # Precedence constraint
                for k in range(self.ES[successor], self.LS[successor] + 1):
                    first_half = self._get_forward_staircase_register(successor,
                                                                      self.ES[successor],
                                                                      k + 1)

                    if first_half is None or k - self.problem.durations[predecessor] + 1 >= self.LS[
                        predecessor] + 1:
                        continue

                    self.sat_model.add_hard_clause(
                        [-first_half,
                         -self.start[predecessor, k - self.problem.durations[predecessor] + 1]])
                else:
                    for k in range(self.LS[successor] - self.problem.durations[predecessor] + 2,
                                   self.LS[predecessor] + 1):
                        self.sat_model.add_hard_clause(
                            [-self._get_forward_staircase_register(successor, self.ES[successor],
                                                                   self.LS[successor] + 1),
                             -self.start[predecessor, k]])

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
                self.register[current_tuple] = self.sat_model.get_new_var()

                # Create constraint for staircase

                # If current tuple is true then the register associated with it must be true
                self.sat_model.add_hard_clause(
                    [-self.start[(job, s)], self.register[current_tuple]])

                if s == start:
                    self.sat_model.add_hard_clause(
                        [self.start[(job, s)], -self.register[current_tuple]])
                else:
                    # Get the previous tuple
                    previous_tuple = tuple(accumulative[:-1])
                    # If previous tuple is true then the current register must be true
                    self.sat_model.add_hard_clause(
                        [-self.register[previous_tuple], self.register[current_tuple]])

                    # Both previous tuple and current variable is false then current tuple must be false
                    self.sat_model.add_hard_clause(
                        [self.register[previous_tuple], self.start[job, s],
                         -self.register[current_tuple]])

                    # Previous tuple and current variable must not be true at the same time
                    self.sat_model.add_hard_clause(
                        [-self.register[previous_tuple], -self.start[(job, s)]])

        return self.register[temp]

    def _resource_constraint(self):
        for t in range(self.makespan):
            for r in range(self.problem.nresources):
                pb_constraint = PBConstraint(self.sat_model, self.problem.capacities[r])
                for i in range(self.problem.njobs):
                    if t in range(self.ES[i], self.LC[i] + 1):
                        pb_constraint.add_term(self.run[(i, t)], self.problem.requests[i][r])
                pb_constraint.encode()

    def _redundant_constraint(self):
        for i in range(self.problem.njobs):
            for c in range(self.EC[i], self.LC[i]):
                self.sat_model.add_hard_clause(
                    [-self.run[(i, c)], self.run[(i, c + 1)],
                     self.start[(i, c - self.problem.durations[i] + 1)]])

    def _consistency_constraint(self):
        for i in range(self.problem.njobs):
            for s in range(self.ES[i], self.LS[i] + 1):  # s in STW(i)
                for t in range(s, s + self.problem.durations[i]):
                    self.sat_model.add_hard_clause(
                        [-self.start[(i, s)], self.run[(i, t)]])
                    self.sat_model.number_of_consistency_clause += 1

    def _start_time_for_job_0(self):
        self.sat_model.add_hard_clause([self.start[(0, 0)]])
