import logging
import os
import os.path
import re
import secrets
import string
import subprocess
import timeit
from enum import Enum, auto
from pathlib import Path
from threading import Timer
from typing import Iterable

from pysat.pb import PBEnc, EncType
from pysat.solvers import Glucose4


def generate_random_filename() -> str:
    """Generate a random filename with 5 alphanumeric characters."""
    characters = string.ascii_letters + string.digits
    random_str = ''.join(secrets.choice(characters) for _ in range(5))
    return random_str


def get_project_root() -> Path:
    """Get the root directory of the project."""
    return Path(__file__).parent.parent


def create_eprime_file(pbs: list[tuple[list[int], list[int], int]]) -> str:
    """
    Create an EPrime file from the given pseudo-Boolean constraints (pbs).
    Args:
        pbs (list[tuple[list[int], list[int], int]]): A list of pseudo-Boolean constraints,
            where each constraint is a tuple of (literals, weights, bound). Example: [([1, 2], [3, 4], 5), ([3, 4], [1, 2], 6)]
    Returns:
        The path to the created EPrime file.
    """
    os.makedirs(os.path.join(get_project_root(), 'eprime'), exist_ok=True)
    file_path = os.path.join(get_project_root(), 'eprime', f'{generate_random_filename()}.eprime')

    clauses = []
    unique_literals = set()

    for pb in pbs:
        if len(pb[0]) == 1 and pb[1][0] == 1 and pb[2] == 1:
            continue

        if sum(pb[1]) <= pb[2]:
            continue

        unique_literals.update(pb[0])
        clause = '+'.join(f"{w}*x{l}" for l, w in zip(pb[0], pb[1]) if w != 0)
        if not clause:
            continue

        clause += f"<={pb[2]}"
        clauses.append(clause)

    with open(file_path, 'a+') as f:
        f.write("language ESSENCE' 1.0\n")
        for literal in unique_literals:
            f.write(f"find\t x{literal}:bool\n")
        f.write("such that\n")
        f.write("/\\\n".join(clauses))

    return file_path



class SOLVER_STATUS(Enum):
    """
    Enum to represent the status of the solver.
    """
    UNSATISFIABLE = auto()
    OPTIMAL = auto()
    SATISFIABLE = auto()
    UNKNOWN = auto()


class SATSolver:
    """
    A warper class for the SAT solver.
    This class is responsible for managing the SAT solver, adding clauses, and solving the SAT problem,
    managing assumptions, and providing statistics.
    """

    def __init__(self):
        """
        Initialize the SAT solver.
        """
        self.__number_of_variables = 0
        self.__solver = None
        self.__assumption = set()
        self.__last_feasible_model = None
        self.__number_of_calls = 0
        self.__temp_clauses = set()

    def create_new_variable(self) -> int:
        """
        Create a new variable in the SAT solver.

        Returns:
            int: The index of the new variable.
        """
        self.__number_of_variables += 1
        return self.__number_of_variables

    def add_clause(self, clause: Iterable[int]):
        """
        Add a clause to the SAT solver.

        Args:
            clause (Iterable[int]): The clause to be added.
        """
        self.__temp_clauses.add(tuple(sorted(clause)))

    def solve(self, time_limit=None) -> bool | None:
        """
        Solve the SAT problem using the current clauses.

        Args:
            time_limit (int): Optional time limit for the solver in seconds. If None, no time limit is applied.
        Returns:
            bool | None: True if the problem is satisfiable, False if it is unsatisfiable,
                         None if the problem could not be solved within the time limit.
        Raises:
            ValueError: If the time limit is less than or equal to 0.
        """
        if time_limit is not None and time_limit <= 0:
            raise ValueError("Time limit must be greater than 0.")

        if self.__solver is None:
            logging.info(
                "Sorting the clauses and initializing the SAT solver with the current clauses..."
            )
            t = timeit.default_timer()
            self.__temp_clauses = sorted(self.__temp_clauses, key=lambda t: (len(t), t))
            self.__solver = Glucose4(bootstrap_with=self.__temp_clauses, use_timer=True, incr=True)

            logging.info(
                f"Finished initializing the SAT solver with {len(self.__temp_clauses)} clauses in "
                f"{round(timeit.default_timer() - t, 5)} seconds."
            )

        logging.info(f"Solving the SAT call {self.__number_of_calls}...")
        self.__number_of_calls += 1
        start = timeit.default_timer()
        self.__solver.clear_interrupt()
        timer = None
        if time_limit is not None:
            def interrupt(s):
                s.interrupt()
                logging.info("Solver interrupted due to timeout.")

            timer = Timer(time_limit, interrupt, [self.__solver])
            timer.start()

        try:
            result = self.__solver.solve_limited(expect_interrupt=True,
                                                 assumptions=self.__assumption)
            if result is not None and result:
                self.__last_feasible_model = self.__solver.get_model()
            return result
        finally:
            if timer:
                timer.cancel()
            logging.info("Finished solving.")
            logging.info(
                f"Solving time for this call: {round(timeit.default_timer() - start, 5)} seconds.")
            logging.info(
                f"Total solving time: {round(self.__solver.time_accum(), 5)} seconds.")

    def get_last_feasible_model(self) -> list[int] | None:
        """
        Get the last feasible model from the SAT solver.

        Returns:
            The last feasible model as a list of integers or None if no model is available.
        """
        return self.__last_feasible_model

    def get_model(self) -> list[int] | None:
        """
        Get the model from the SAT solver if it is satisfiable.
        Returns:
             The model as a list of integers or None if the problem is unsatisfiable.
        """
        return self.__solver.get_model()

    def add_assumption(self, assumption: int):
        """
        Add an assumption to the SAT solver.
        Args:
             assumption (int): The assumption to be added.
        """
        self.__assumption.add(assumption)

    def clear_interrupt(self):
        """
        Clear any interrupt set on the SAT solver.
        This method is used to reset the interrupt state of the solver, allowing it to continue solving
        if it was interrupted.

        """
        self.__solver.clear_interrupt()

    def get_statistics(self) -> dict[str, int | float]:
        """
        Get the statistics of the SAT solver.

        Returns:
            dict[str, int | float]: A dictionary containing the statistics of the SAT solver.

            This dictionary keys include:
                - "variables": The number of variables in the SAT solver.
                - "clauses": The number of clauses in the SAT solver.
                - "hard_clauses": The number of hard clauses in the SAT solver.
                - "soft_clauses": The number of soft clauses in the SAT solver (always 0 for this class).
                - "total_solving_time": The total time spent solving the SAT problem.
                - "number_of_calls": The number of times the solver has been called.
        """
        return {
            "variables": self.__number_of_variables,
            "clauses": len(self.__temp_clauses),
            "hard_clauses": len(self.__temp_clauses),
            "soft_clauses": 0,
            "total_solving_time": round(self.__solver.time_accum(), 5),
            "number_of_calls": self.__number_of_calls
        }

    def add_at_most_k(self, literals: list[int], weights: list[int], k: int):
        """
        Add a weighted at most k constraint to the SAT solver.
        :param literals: List of literals involved in the constraint.
        :param weights: List of weights corresponding to each literal.
        :param k: The bound for the weighted at most k constraint.
        """
        cnf = PBEnc.leq(lits=literals, weights=weights, bound=k,
                        top_id=self.__number_of_variables, encoding=EncType.bdd).clauses

        if not cnf:
            return

        new_variable_max_index = -1
        for clause in cnf:
            for var in clause:
                if abs(var) > new_variable_max_index:
                    new_variable_max_index = abs(var)
        if new_variable_max_index == -1:
            return

        self.__number_of_variables = max(new_variable_max_index, self.__number_of_variables)
        for clause in cnf:
            self.__temp_clauses.add(tuple(sorted(clause)))

    def __parse_eprime_file(self, file_path: str):
        literals_mapping = {}
        auxiliary_mapping = {}
        literal_found = set()

        with open(file_path, "r") as f:
            while True:
                line = f.readline()

                if not line:
                    break  # EOF

                if line.startswith('p'):
                    continue

                # Check if the line is an "Encoding variable" comment
                if line.startswith("c Encoding variable:"):
                    var_line = line
                    sat_line = f.readline()
                    var_match = re.search(r'x\d+', var_line)
                    sat_match = re.search(r'\d+', sat_line)
                    if var_match and sat_match:
                        literals_mapping[int(sat_match.group())] = int(var_match.group()[1:])
                        literal_found.add(int(var_match.group()[1:]))

                elif line.startswith("c"):
                    continue

                else:
                    raw = list(map(int, line.split(' ')))

                    clause = []

                    for var in raw:
                        if var == 0:
                            continue

                        if abs(var) in literals_mapping:
                            clause.append(
                                literals_mapping[var] if var > 0 else -literals_mapping[abs(var)])

                        else:
                            if abs(var) not in auxiliary_mapping:
                                auxiliary_mapping[abs(var)] = self.create_new_variable()

                            clause.append(
                                auxiliary_mapping[var] if var > 0 else -auxiliary_mapping[abs(var)])

                    self.add_clause(clause)

    def add_pb_clauses(self, pbs: list[tuple[list[int], list[int], int]]):
        """
        Add pseudo-Boolean constraints to the SAT solver.
        This method creates an EPrime file from the given pseudo-Boolean constraints and
        uses the Savile Row tool to convert it into a DIMACS file, which is then parsed
        to add clauses to the SAT solver.

        Args:
            pbs (list[tuple[list[int], list[int], int]]): A list of pseudo-Boolean constraints,
                where each constraint is a tuple of (literals, weights, bound).
                Example: [([1, 2], [3, 4], 5), ([3, 4], [1, 2], 6)]

        Returns:

        """
        eprime_path = create_eprime_file(pbs)

        os.makedirs(os.path.join(get_project_root(), 'dimacs'), exist_ok=True)
        output_file_path = os.path.join(get_project_root(), 'dimacs',
                                        f"{generate_random_filename()}.dimacs")

        command = (f"{get_project_root()}/bin/savilerow/savilerow {eprime_path} "
                   f"-sat -sat-pb-mdd -amo-detect -out-sat {output_file_path} ")

        process = subprocess.Popen(command, shell=True, stdout=subprocess.DEVNULL)
        process.wait()

        self.__parse_eprime_file(output_file_path)


class MaxSATSolver:
    """
    A warper class for the MaxSAT solver.
    This class is responsible for managing the MaxSAT solver, adding hard and soft clauses,
    solving the MaxSAT problem, and providing statistics.
    """

    def __init__(self):
        """
        Initialize the MaxSAT solver.
        """
        self.__output_file = None
        self.__return_code = None
        self.__time_used = 0
        self.__number_of_variables = 0
        self.__soft_clauses: set[tuple[tuple[int, ...], int]] = set()
        self.__hard_clauses = set()
        self.__model = None

    def create_new_variable(self) -> int:
        """
        Create a new variable in the SAT solver.
        :return: The index of the new variable.
        """
        self.__number_of_variables += 1
        return self.__number_of_variables

    def add_hard_clause(self, clause: list[int]):
        """
        Add a clause to the SAT solver.
        :param clause: The clause to be added.
        """
        self.__hard_clauses.add(tuple(sorted(clause)))

    def add_soft_clause(self, clause: list[int], weight: int):
        """
        Add a soft clause to the MaxSAT solver.
        :param clause: The clause to be added.
        :param weight: The weight of the soft clause.
        :raises ValueError: If the weight of the soft clause is less than or equal to 0.
        """
        if weight <= 0:
            raise ValueError("Weight of a soft clause must be greater than 0.")
        self.__soft_clauses.add((tuple(sorted(clause)), weight))

    def solve(self, time_limit=None) -> SOLVER_STATUS:
        """
        Solve the SAT problem using the current clauses.
        :return: The status of the solver after attempting to solve the problem.
        """
        project_root = str(get_project_root())
        os.makedirs(project_root + '/wcnf', exist_ok=True)
        os.makedirs(project_root + '/out', exist_ok=True)

        file_name = generate_random_filename() + ".wcnf"
        self.__output_file = project_root + '/out/' + file_name + '.out'
        # Export the SAT model to wcnf folder
        self.export(project_root + '/wcnf/' + file_name)

        # Solve the problem using the MaxSAT solver
        logging.info("Solving the MaxSAT problem...")
        start = timeit.default_timer()
        if time_limit is not None:
            command = (
                f"timeout -s SIGTERM {time_limit}s {get_project_root()}/bin/tt-open-wbo-inc-Glucose4_1_static "
                f"wcnf/{file_name} > {self.__output_file}"
            )
        else:
            command = (
                f"{get_project_root()}/bin/tt-open-wbo-inc-Glucose4_1_static wcnf/{file_name} > {self.__output_file}"
            )

        process = subprocess.Popen(command, shell=True)
        process.wait()

        self.__time_used = timeit.default_timer() - start

        logging.info(f"Finished solving the problem.")
        logging.info(f"Total solving time: {round(self.__time_used, 5)} seconds.")

        logging.info(f"Getting the result from the output file: {self.__output_file}")
        start = timeit.default_timer()
        last_s_line = ""

        with open(self.__output_file, 'r') as f:
            for line in f:
                if line.startswith('s '):
                    # Keep only the last 's' line
                    last_s_line = line[2:].strip()

        logging.info(
            f"Finished getting the result from the output file: {self.__output_file}")
        logging.info(
            f"Time taken to read the output file for solver's status: {round(timeit.default_timer() - start, 5)} seconds.")

        if last_s_line == "UNSATISFIABLE":
            self.__return_code = SOLVER_STATUS.UNSATISFIABLE
        elif last_s_line == "SATISFIABLE":
            self.__return_code = SOLVER_STATUS.SATISFIABLE
        elif last_s_line == "UNKNOWN":
            self.__return_code = SOLVER_STATUS.UNKNOWN
        else:
            self.__return_code = SOLVER_STATUS.OPTIMAL

        self.get_model()

        return self.__return_code

    def get_model(self) -> list[int] | None:
        """
        Get the model from the MaxSAT solver if it is satisfiable.
        :return: The model as a list of integers or None if the problem is unsatisfiable.
        """
        if (self.__return_code == SOLVER_STATUS.UNSATISFIABLE or
                self.__return_code == SOLVER_STATUS.UNKNOWN):
            return None

        if self.__model is not None:
            return self.__model

        logging.info('Getting the model from the output file: ' + self.__output_file)
        start = timeit.default_timer()
        last_v_line = ""
        with open(self.__output_file, 'r') as f:
            for line in f:
                if line.startswith('v '):
                    # Keep only the last 'v' line
                    last_v_line = line[2:].strip()

        # Use the last "v" line as the solution
        cleaned = last_v_line.replace('v', '').strip()

        self.__model = [int(d) for d in cleaned]
        logging.info(f"Finished getting the model from the output file: {self.__output_file}")
        logging.info(
            f"Time taken to read the output file for model: {round(timeit.default_timer() - start, 5)} seconds.")
        return self.__model

    def add_at_most_k(self, literals: list[int], weights: list[int], k: int):
        """
        Add a weighted at most k constraint to the SAT solver.
        :param literals: List of literals involved in the constraint.
        :param weights: List of weights corresponding to each literal.
        :param k: The bound for the weighted at most k constraint.
        """
        cnf = PBEnc.leq(lits=literals, weights=weights, bound=k,
                        top_id=self.__number_of_variables, encoding=EncType.bdd).clauses

        if not cnf:
            return

        new_variable_max_index = -1
        for clause in cnf:
            for var in clause:
                if abs(var) > new_variable_max_index:
                    new_variable_max_index = abs(var)
        if new_variable_max_index == -1:
            return

        self.__number_of_variables = max(new_variable_max_index, self.__number_of_variables)
        for clause in cnf:
            self.add_hard_clause(clause)

    def export(self, filename: str):
        """
        Export the clauses to a WCNF file.
        :param filename: The name of the file to export to.
        """
        logging.info(f"Exporting the MaxSAT model to {filename}...")
        start = timeit.default_timer()
        with open(filename, 'a+') as f:
            # Write hard clauses first
            for clause in self.__hard_clauses:
                f.write("h " + " ".join(map(str, clause)) + " 0\n")

            # Write soft clauses with their weight
            for clause, weight in self.__soft_clauses:
                f.write(f"{weight} " + " ".join(map(str, clause)) + " 0\n")
        logging.info(f"Finished exporting the MaxSAT model to {filename}.")
        logging.info(
            f"Time taken to export: {round(timeit.default_timer() - start, 5)} seconds.")

    def get_statistics(self) -> dict[str, int | float]:
        """
        Get the statistics of the SAT solver.
        :return: A dictionary containing the statistics of the SAT solver."""
        return {
            "variables": self.__number_of_variables,
            "clauses": len(self.__soft_clauses) + len(self.__hard_clauses),
            "hard_clauses": len(self.__hard_clauses),
            "soft_clauses": len(self.__soft_clauses),
            "total_solving_time": round(self.__time_used, 5)
        }
