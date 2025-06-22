import logging
import os
import os.path
import secrets
import string
import subprocess
import timeit
from enum import Enum, auto
from pathlib import Path
from threading import Timer

from pysat.pb import PBEnc, EncType
from pysat.solvers import Glucose4


def generate_random_filename():
    """Generate a random filename with 5 alphanumeric characters."""
    characters = string.ascii_letters + string.digits
    random_str = ''.join(secrets.choice(characters) for _ in range(5))
    return random_str


def get_project_root() -> Path:
    """Get the root directory of the project."""
    return Path(__file__).parent.parent


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
        self.__number_of_clauses = 0
        self.__solver = Glucose4(use_timer=True, incr=True)
        self.__assumption = set()
        self.__last_feasible_model = None
        self.__number_of_calls = 0

    def create_new_variable(self) -> int:
        """
        Create a new variable in the SAT solver.
        :return: The index of the new variable.
        """
        self.__number_of_variables += 1
        return self.__number_of_variables

    def add_clause(self, clause: list[int]):
        """
        Add a clause to the SAT solver.
        :param clause: The clause to be added.
        """
        self.__number_of_clauses += 1
        self.__solver.add_clause(clause)

    def solve(self, time_limit=None) -> bool | None:
        """
        Solve the SAT problem using the current clauses.
        :return: The status of the solver after attempting to solve the problem.
        """
        if time_limit is not None and time_limit <= 0:
            return None
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
        :return: The last feasible model as a list of integers or None if no model is available.
        """
        return self.__last_feasible_model

    def get_model(self) -> list[int] | None:
        """
        Get the model from the SAT solver if it is satisfiable.
        :return: The model as a list of integers or None if the problem is unsatisfiable.
        """
        return self.__solver.get_model()

    def add_assumption(self, assumption: int):
        """
        Add an assumption to the SAT solver.
        :param assumption: The assumption to be added.
        """
        self.__assumption.add(assumption)

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
            self.add_clause(clause)

    def clear_interrupt(self):
        self.__solver.clear_interrupt()

    def get_statistics(self) -> dict[str, int | float]:
        """
        Get the statistics of the SAT solver.
        :return: A dictionary containing the statistics of the SAT solver."""
        return {
            "variables": self.__number_of_variables,
            "clauses": self.__number_of_clauses,
            "hard_clauses": self.__number_of_clauses,
            "soft_clauses": 0,
            "total_solving_time": round(self.__solver.time_accum(), 5),
            "number_of_calls": self.__number_of_calls
        }


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
        self.__number_of_clauses = 0
        self.__number_of_soft_clauses = 0
        self.__number_of_hard_clauses = 0
        self.__soft_clauses: list[tuple[list[int], int]] = []
        self.__hard_clauses = []
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
        self.__number_of_clauses += 1
        self.__number_of_hard_clauses += 1
        self.__hard_clauses.append(clause)

    def add_soft_clause(self, clause: list[int], weight: int):
        """
        Add a soft clause to the MaxSAT solver.
        :param clause: The clause to be added.
        :param weight: The weight of the soft clause.
        :raises ValueError: If the weight of the soft clause is less than or equal to 0.
        """
        if weight <= 0:
            raise ValueError("Weight of a soft clause must be greater than 0.")
        self.__number_of_clauses += 1
        self.__number_of_soft_clauses += 1
        self.__soft_clauses.append((clause, weight))
        self.__number_of_soft_clauses += 1

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
            "clauses": self.__number_of_clauses,
            "hard_clauses": self.__number_of_hard_clauses,
            "soft_clauses": self.__number_of_soft_clauses,
            "total_solving_time": round(self.__time_used, 5)
        }
