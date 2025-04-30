from enum import Enum

from pysat.solvers import Glucose4


class NUMBER_OF_LITERAL(Enum):
    """
    Enum to represent the number of literals in a clause.
    """
    ZERO = 0
    ONE = 1
    TWO = 2
    THREE = 3
    FOUR = 4
    FIVE_TO_TEN = 5
    MORE_THAN_TEN = 11


class SATModel:
    """
    A wrapper class for the SAT solver.
    It manages the number of variables, clauses, and literals.
    """

    def __init__(self):
        """
        Initialize the SAT model.
        """
        self.__number_of_variables = 0
        self.__number_of_clauses = 0
        self.__number_of_PB_clauses = 0
        self.__number_of_consistency_clauses = 0
        self.__solver = Glucose4(use_timer=True, incr=True)
        self.__number_of_literals = {NUMBER_OF_LITERAL.ZERO.value: 0,
                                     NUMBER_OF_LITERAL.ONE.value: 0,
                                     NUMBER_OF_LITERAL.TWO.value: 0,
                                     NUMBER_OF_LITERAL.THREE.value: 0,
                                     NUMBER_OF_LITERAL.FOUR.value: 0,
                                     NUMBER_OF_LITERAL.FIVE_TO_TEN.value: 0,
                                     NUMBER_OF_LITERAL.MORE_THAN_TEN.value: 0}

    @property
    def number_of_variables(self) -> int:
        """
        Get the number of variables in the SAT solver.
        :return: The number of variables.
        :rtype: int
        """
        return self.__number_of_variables

    @number_of_variables.setter
    def number_of_variables(self, value: int):
        """
        Set the number of variables in the SAT solver.
        :param value: The number of variables.
        :type value: int
        """
        self.__number_of_variables = value

    @property
    def number_of_clauses(self) -> int:
        """
        Get the number of clauses in the SAT solver.
        :return: The number of clauses.
        :rtype: int
        """
        return self.__number_of_clauses

    @property
    def number_of_PB_clauses(self) -> int:
        """
        Get the number of PB clauses in the SAT solver.
        :return: The number of PB clauses.
        :rtype: int
        """
        return self.__number_of_PB_clauses

    @number_of_PB_clauses.setter
    def number_of_PB_clauses(self, value: int):
        """
        Set the number of PB clauses in the SAT solver.
        :param value: The number of PB clauses.
        :type value: int
        """
        self.__number_of_PB_clauses = value

    @property
    def number_of_consistency_clauses(self) -> int:
        """
        Get the number of consistency clauses in the SAT solver.
        :return: The number of consistency clauses.
        :rtype: int
        """
        return self.__number_of_consistency_clauses

    @number_of_consistency_clauses.setter
    def number_of_consistency_clauses(self, value: int):
        """
        Set the number of consistency clauses in the SAT solver.
        :param value: The number of consistency clauses.
        :type value: int
        """
        self.__number_of_consistency_clauses = value

    @property
    def number_of_literals(self) -> dict[int, int]:
        """
        Get the number of literals in the SAT solver.
        :return: A dictionary with the count of literals.
        :rtype: dict[int, int]
        """
        return self.__number_of_literals

    @property
    def solver(self) -> Glucose4:
        """
        Get the SAT solver instance.
        :return: The SAT solver instance.
        :rtype: Glucose4
        """
        return self.__solver

    def create_new_variable(self) -> int:
        """
        Create a new variable in the SAT solver.
        :return: The index of the new variable.
        :rtype: int
        """
        self.__number_of_variables += 1
        return self.__number_of_variables

    def add_clause(self, clause: list[int]):
        """
        Add a clause to the SAT solver.
        :param clause: The clause to be added.
        :type clause: list[int]
        """
        self.__solver.add_clause(clause)
        self.__number_of_clauses += 1
        if len(clause) > 10:
            self.__number_of_literals[NUMBER_OF_LITERAL.MORE_THAN_TEN.value] += 1
        elif len(clause) > 4:
            self.__number_of_literals[NUMBER_OF_LITERAL.FIVE_TO_TEN.value] += 1
        else:
            self.__number_of_literals[len(clause)] += 1
