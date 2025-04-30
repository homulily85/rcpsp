import os
import secrets
import shutil
import string
import subprocess
import timeit

from encoder.model.status import SOLVER_STATUS
from encoder.rcpsp_solver import RCPSPSolver


class LIASolver(RCPSPSolver):
    """
    Class for solving the Resource-Constrained Project Scheduling Problem (RCPSP) using the
    LIA solver from M. Bofill et al. “SMT encodings for resource-constrained project scheduling problems"
    (2020).
    """

    def __init__(self, input_file: str,
                 timeout: int = None,
                 enable_verify: bool = False):
        """
        Initialize the LIA solver with the input file, timeout, and verification flag.
        :param input_file: Path to the input file.
        :type input_file: str
        :param timeout: Timeout for the solver in seconds.
        :type timeout: int|None
        :param enable_verify: Flag to enable verification of the solution.
        :type enable_verify: bool
        """
        super().__init__(input_file, timeout, enable_verify)
        self.__return_code = None
        self.__output_file = None

    @property
    def makespan(self) -> int:
        if self.__return_code in [
            SOLVER_STATUS.UNSATISFIABLE.value, SOLVER_STATUS.UNKNOWN.value]:
            raise Exception("No solution found")
        return self.solution[-1]

    @staticmethod
    def __generate_random_filename():
        """
        Generate a random filename for the output file.
        :return: A random filename.
        :rtype: str
        """
        characters = string.ascii_letters + string.digits
        random_str = ''.join(secrets.choice(characters) for _ in range(5))
        return f"{random_str}"

    def solve(self):
        # Check if wcnf folder exists
        if os.path.exists('out'):
            shutil.rmtree('out')

        os.makedirs('out')

        file_name = self.__generate_random_filename()
        self.__output_file = 'out/' + file_name + '.out'

        # Solve the problem using the original LIA solver
        start = timeit.default_timer()
        if self._timeout is not None:
            command = (
                f"timeout -s SIGTERM {self._timeout}s "
                f"env LD_LIBRARY_PATH=./bin:./bin/yices-2.6.0/lib ./bin/mrcpsp2smt {self._input_file} "
                f"--amopb=lia --pb=lia > {self.__output_file}"
            )
        else:
            command = (
                f"env LD_LIBRARY_PATH=./bin:./bin/yices-2.6.0/lib ./bin/mrcpsp2smt {self._input_file} "
                f"--amopb=lia --pb=lia > {self.__output_file}"
            )

        process = subprocess.Popen(command, shell=True)
        process.wait()

        self._time_used = timeit.default_timer() - start

        # Check if the solution is optimal by reading the output file
        with open(self.__output_file, 'r') as f:
            file_content = f.read()
            if "OPTIMUM FOUND" in file_content:
                self.__return_code = SOLVER_STATUS.OPTIMUM
            else:
                self.__return_code = SOLVER_STATUS.SATISFIABLE

        return self.__return_code

    @property
    def solution(self) -> list[int]:
        if self.__return_code in [
            SOLVER_STATUS.UNSATISFIABLE.value, SOLVER_STATUS.UNKNOWN.value]:
            raise Exception("No solution found")

        # Parse the solution from the output file
        if self._solution is None:
            self._solution = self.__parse_solution()
        return self._solution

    def __parse_solution(self) -> list[int]:
        """
        Parse the _solution from the output file and extract start times for each activity.
        :return: A list of start times for each activity.
        :rtype: list[int]
        """
        # Find the last line starting with 'v'
        last_v_line = ""
        with open(self.__output_file, 'r') as f:
            for line in f:
                if line.startswith('v '):
                    # Keep only the last 'v' line
                    last_v_line = line[2:].strip()

        if not last_v_line:
            raise Exception("No solution found in output file")

        # Parse the S_i:t values
        parts = last_v_line.split(';')

        # Find all S_i:t pairs and store them
        activities = {}
        for part in parts:
            part = part.strip()
            if part.startswith('S_'):
                # Extract activity index and start time
                activity_info = part.split(':')
                if len(activity_info) == 2:
                    activity_index = int(activity_info[0][2:])  # Remove 'S_' prefix
                    start_time = int(activity_info[1])
                    activities[activity_index] = start_time

        # Determine the number of activities
        if activities:
            num_activities = max(activities.keys()) + 1
        else:
            raise Exception("No activity start times found in solution")

        # Create result list
        result = [-1] * num_activities

        # Fill in start times
        for idx, time in activities.items():
            result[idx] = time

        return result
