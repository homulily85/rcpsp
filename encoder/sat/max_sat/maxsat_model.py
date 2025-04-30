from encoder.sat.number_of_literals import NUMBER_OF_LITERALS


class MaxSATModel:
    """
    A class representing a MaxSAT model.
    """

    def __init__(self):
        """
        Initialize the MaxSAT model.
        """
        self.__number_of_variables = 0
        self.__number_of_clauses = 0
        self.__number_of_PB_clauses = 0
        self.__number_of_consistency_clauses = 0
        self.__number_of_literals = {
            NUMBER_OF_LITERALS.ZERO: 0,
            NUMBER_OF_LITERALS.ONE: 0,
            NUMBER_OF_LITERALS.TWO: 0,
            NUMBER_OF_LITERALS.THREE: 0,
            NUMBER_OF_LITERALS.FOUR: 0,
            NUMBER_OF_LITERALS.FIVE_TO_TEN: 0,
            NUMBER_OF_LITERALS.MORE_THAN_TEN: 0,
        }
        self.__number_of_soft_clauses = 0
        self.__number_of_hard_clauses = 0
        self.__soft_clauses: list[tuple[list[int], int]] = []
        self.__hard_clauses = []

    @property
    def number_of_variables(self) -> int:
        """
        Get the number of variables in the MaxSAT model.
        :return: The number of variables.
        :rtype: int
        """
        return self.__number_of_variables

    @number_of_variables.setter
    def number_of_variables(self, value: int):
        """
        Set the number of variables in the MaxSAT model.
        :param value: The number of variables.
        :type value: int
        """
        self.__number_of_variables = value

    @property
    def number_of_clauses(self) -> int:
        """
        Get the number of clauses in the MaxSAT model.
        :return: The number of clauses.
        :rtype: int
        """
        return self.__number_of_clauses

    @property
    def number_of_soft_clauses(self) -> int:
        """
        Get the number of soft clauses in the MaxSAT model.
        :return: The number of soft clauses.
        :rtype: int
        """
        return self.__number_of_soft_clauses

    @property
    def number_of_hard_clauses(self) -> int:
        """
        Get the number of hard clauses in the MaxSAT model.
        :return: The number of hard clauses.
        :rtype: int
        """
        return self.__number_of_hard_clauses

    @property
    def number_of_pb_clauses(self) -> int:
        """
        Get the number of PB clauses in the MaxSAT model.
        :return: The number of PB clauses.
        :rtype: int
        """
        return self.__number_of_PB_clauses
    @number_of_pb_clauses.setter
    def number_of_pb_clauses(self, value: int):
        """
        Set the number of PB clauses in the MaxSAT model.
        :param value: The number of PB clauses.
        :type value: int
        """
        self.__number_of_PB_clauses = value

    @property
    def number_of_consistency_clauses(self) -> int:
        """
        Get the number of consistency clauses in the MaxSAT model.
        :return: The number of consistency clauses.
        :rtype: int
        """
        return self.__number_of_consistency_clauses
    @number_of_consistency_clauses.setter
    def number_of_consistency_clauses(self, value: int):
        """
        Set the number of consistency clauses in the MaxSAT model.
        :param value: The number of consistency clauses.
        :type value: int
        """
        self.__number_of_consistency_clauses = value

    @property
    def number_of_literals(self) -> dict[NUMBER_OF_LITERALS, int]:
        """
        Get the number of literals in the MaxSAT model.
        :return: The number of literals.
        :rtype: dict[NUMBER_OF_LITERALS, int]
        """
        return self.__number_of_literals

    def create_new_var(self) -> int:
        """
        Create a new variable for the MaxSAT model.
        :return: The new variable ID.
        :rtype: int
        """
        self.__number_of_variables += 1
        return self.__number_of_variables

    def __modify_statistic(self, clause: list[int]):
        """
        Modify the statistics of the MaxSAT model based on the clause.
        :param clause: The clause to modify the statistics for.
        :type clause: list[int]
        """
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

    def add_soft_clause(self, clause: list[int], weight: int):
        """
        Add a soft clause to the MaxSAT model.
        :param clause: The clause to be added.
        :type clause: list[int]
        :param weight: The weight of the clause.
        :type weight: int
        """
        self.__soft_clauses.append((clause, weight))
        self.__number_of_soft_clauses += 1
        self.__modify_statistic(clause)

    def add_hard_clause(self, clause: list[int]):
        """
        Add a hard clause to the MaxSAT model.
        :param clause: The clause to be added.
        :type clause: list[int]
        """
        self.__hard_clauses.append(clause)
        self.__number_of_hard_clauses += 1
        self.__modify_statistic(clause)

    def export(self, filename: str):
        """
        Export the MaxSAT model to a WCNF file.
        :param filename: The name of the file to export to.
        """
        with open(filename, 'a+') as f:
            # Write header information as comments
            f.write(f"c WCNF file generated by MaxSATModel\n")
            f.write(f"c Number of variables: {self.__number_of_variables}\n")
            f.write(f"c Number of hard clauses: {self.__number_of_hard_clauses}\n")
            f.write(f"c Number of soft clauses: {self.__number_of_soft_clauses}\n")

            # Write hard clauses first
            for clause in self.__hard_clauses:
                f.write("h " + " ".join(map(str, clause)) + " 0\n")

            # Write soft clauses with their weights
            for clause, weight in self.__soft_clauses:
                f.write(f"{weight} " + " ".join(map(str, clause)) + " 0\n")
