from enum import Enum, auto
from threading import Timer
from typing import Iterable

from pysat.pb import PBEnc, EncType
from pysat.solvers import Glucose4


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
            self.__temp_clauses = sorted(self.__temp_clauses, key=lambda t: (len(t), t))
            self.__solver = Glucose4(bootstrap_with=self.__temp_clauses, use_timer=True, incr=True)

        self.__number_of_calls += 1
        self.__solver.clear_interrupt()
        timer = None
        if time_limit is not None:
            def interrupt(s):
                s.interrupt()

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

            This dictionary keys include:CSV
                - "variables": The number of variables in the SAT solver.
                - "clauses": The number of clauses in the SAT solver.
                - "total_solving_time": The total time spent solving the SAT problem.
                - "number_of_calls": The number of times the solver has been called.
        """
        return {
            "variables": self.__number_of_variables,
            "clauses": len(self.__temp_clauses),
            "total_solving_time": self.__solver.time_accum(),
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
