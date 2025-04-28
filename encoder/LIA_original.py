import os
import secrets
import shutil
import string
import subprocess
import timeit

from encoder.sat.max_sat.MaxSAT_encoder import SOLVER_STATUS


class OriginalLIA:
    def __init__(self, input_file: str, time_out: int = None, enable_verify: bool = False):
        self.input_file = input_file
        self.time_out = time_out
        self.enable_verify = enable_verify
        self.time_used = 0
        self.solution = None
        self.return_code = None
        self.output_file = None

    def encode(self):
        pass

    def get_makespan(self) -> int:
        """Get the makespan of the solution."""
        if self.return_code in [
            SOLVER_STATUS.UNSATISFIABLE.value, SOLVER_STATUS.UNKNOWN.value]:
            raise Exception("No solution found")

        return self.get_solution()[-1]

    @staticmethod
    def _generate_random_filename():
        characters = string.ascii_letters + string.digits
        random_str = ''.join(secrets.choice(characters) for _ in range(5))
        return f"{random_str}"

    def solve(self):
        # Check if wcnf folder exists
        if os.path.exists('out'):
            shutil.rmtree('out')

        os.makedirs('out')

        file_name = self._generate_random_filename()
        self.output_file = 'out/' + file_name + '.out'

        # Solve the problem using the original LIA solver
        start = timeit.default_timer()
        if self.time_out is not None:
            command = (
                f"timeout -s SIGTERM {self.time_out}s "
                f"env LD_LIBRARY_PATH=./bin ./bin/mrcpsp2smt {self.input_file} "
                f"--amopb=lia --pb=lia> {self.output_file}"
            )
        else:
            command = (
                f"env LD_LIBRARY_PATH=./bin ./bin/mrcpsp2smt {self.input_file} "
                f"--amopb=lia --pb=lia > {self.output_file}"
            )

        process = subprocess.Popen(command, shell=True)
        process.wait()

        self.time_used = timeit.default_timer() - start

        # Check if the solution is optimal by reading the output file
        with open(self.output_file, 'r') as f:
            file_content = f.read()
            if "OPTIMUM FOUND" in file_content:
                self.return_code = SOLVER_STATUS.OPTIMUM
            else:
                self.return_code = SOLVER_STATUS.SATISFIABLE

        return self.return_code

    def get_solution(self) -> list[int]:
        if self.return_code in [
            SOLVER_STATUS.UNSATISFIABLE.value, SOLVER_STATUS.UNKNOWN.value]:
            raise Exception("No solution found")

        # Parse the solution from the output file
        if self.solution is None:
            self.solution = self._parse_solution()
        return self.solution

    def _parse_solution(self) -> list[int]:
        """
        Parse the solution from the output file and extract start times for each activity.

        Returns:
            list[int]: A list where index i contains the start time of activity i.
        """
        # Find the last line starting with 'v'
        last_v_line = ""
        with open(self.output_file, 'r') as f:
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

    def verify(self):
        raise NotImplementedError()
