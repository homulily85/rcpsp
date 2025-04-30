from enum import Enum


class NUMBER_OF_LITERALS(Enum):
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
