from pysat.solvers import Glucose4

from encoder.sat.number_of_literals import NUMBER_OF_LITERALS


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
        self.__number_of_pb_clauses = 0
        self.__number_of_consistency_clauses = 0
        self.__solver = Glucose4(use_timer=True, incr=True)
        self.__number_of_literals = {NUMBER_OF_LITERALS.ZERO: 0,
                                     NUMBER_OF_LITERALS.ONE: 0,
                                     NUMBER_OF_LITERALS.TWO: 0,
                                     NUMBER_OF_LITERALS.THREE: 0,
                                     NUMBER_OF_LITERALS.FOUR: 0,
                                     NUMBER_OF_LITERALS.FIVE_TO_TEN: 0,
                                     NUMBER_OF_LITERALS.MORE_THAN_TEN: 0}

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
    def number_of_pb_clauses(self) -> int:
        """
        Get the number of PB clauses in the SAT solver.
        :return: The number of PB clauses.
        :rtype: int
        """
        return self.__number_of_pb_clauses

    @number_of_pb_clauses.setter
    def number_of_pb_clauses(self, value: int):
        """
        Set the number of PB clauses in the SAT solver.
        :param value: The number of PB clauses.
        :type value: int
        """
        self.__number_of_pb_clauses = value

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
    def number_of_literals(self) -> dict[NUMBER_OF_LITERALS, int]:
        """
        Get the number of literals in the SAT solver.
        :return: A dictionary with the count of literals.
        :rtype: dict[NUMBER_OF_LITERALS, int]
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

        match len(clause):
            case 0:
                self.__number_of_literals[NUMBER_OF_LITERALS.ZERO] += 1
            case 1:
                self.__number_of_literals[NUMBER_OF_LITERALS.ONE] += 1
            case 2:
                self.__number_of_literals[NUMBER_OF_LITERALS.TWO] += 1
            case 3:
                self.__number_of_literals[NUMBER_OF_LITERALS.THREE] += 1
            case 4:
                self.__number_of_literals[NUMBER_OF_LITERALS.FOUR] += 1
            case _ if len(clause) <= 10:
                self.__number_of_literals[NUMBER_OF_LITERALS.FIVE_TO_TEN] += 1
            case _:
                self.__number_of_literals[NUMBER_OF_LITERALS.MORE_THAN_TEN] += 1
