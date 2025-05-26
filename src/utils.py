import datetime
import os
import timeit
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
from pandas import DataFrame


def get_project_root() -> Path:
    return Path(__file__).parent.parent


class SimpleLoggerMetaclass(type):
    """A metaclass for the SimpleLogger that ensures only one instance exists."""

    __instance = {}
    __lock: Lock = Lock()

    def __call__(cls, *args, **kwargs):
        with cls.__lock:
            if cls not in cls.__instance:
                cls.__instance[cls] = super().__call__(*args, **kwargs)
        return cls.__instance[cls]


class SimpleLogger(metaclass=SimpleLoggerMetaclass):
    """A simple logger class that writes messages to a log file and optionally prints them to the console."""

    def __init__(self):
        """
        Initialize the Logger.
        """
        self.__file_name = None
        self.__verbose = False

        # Create log directory if it doesn't exist
        os.makedirs(f'{get_project_root()}/log', exist_ok=True)

    def log(self, message: str):
        """Write a message to the log file and optionally print to console.
        :param message: The message to log.
        :raises ValueError: If the log file name is not set."""
        if self.__file_name is None:
            raise ValueError(
                "Log file name is not set. Please set the file_name property before logging.")
        timestamp = datetime.datetime.now()
        with open(self.__file_name, "a+") as f:
            f.write(f'[{timestamp}] {message}\n')
        if self.__verbose:
            print(f'[{timestamp}] {message}', flush=True)

    @property
    def verbose(self) -> bool:
        """Get the verbosity status.
        :return: True if verbose mode is on, False otherwise."""
        return self.__verbose

    @verbose.setter
    def verbose(self, value: bool):
        """Set the verbosity status.
        :param value: True to enable verbose mode, False to disable it."""
        self.__verbose = value

    @property
    def file_name(self) -> str:
        """Get the name of the log file."""
        return self.__file_name

    @file_name.setter
    def file_name(self, value: str):
        """Set the name of the log file.
        :param value: The name of the log file.
        :raises ValueError: If the log file name is already set.
        """
        if self.__file_name is not None:
            raise ValueError("Log file name is already set.")
        self.__file_name = f'{get_project_root()}/log/{value}'

class ResultManagerMetaclass(type):
    """A metaclass for the ResultManager that ensures only one instance exists."""

    __instance = {}
    __lock: Lock = Lock()

    def __call__(cls, *args, **kwargs):
        with cls.__lock:
            if cls not in cls.__instance:
                cls.__instance[cls] = super().__call__(*args, **kwargs)
        return cls.__instance[cls]

class ResultManager(metaclass=ResultManagerMetaclass):
    """Manages statistic output and solution files."""

    def __init__(self):
        """
        Initialize the ResultManager.
        """
        self.__file_name = None
        self.__export_solution = None

        # Create result directory
        os.makedirs(f'{get_project_root()}/statistic', exist_ok=True)
        os.makedirs(f'{get_project_root()}/solution', exist_ok=True)

        self.__statistic = DataFrame(columns=np.array([
            'file_name',
            'lower_bound',
            'upper_bound',
            'variables',
            'clauses',
            'soft_clauses',
            'hard_constraint_clauses',
            'consistency_clauses',
            'pb_clauses',
            'zero_literals',
            'one_literals',
            'two_literals',
            'three_literals',
            'four_literals',
            'five_to_ten_literals',
            'more_than_ten_literals',
            'feasible',
            'makespan',
            'total_solving_time',
            'optimized',
            'timeout'
        ]))

        self.__solution: dict[str, Any] = {}

    @property
    def file_name(self) -> str:
        """Get the file name for statistics and solution."""
        return self.__file_name

    @file_name.setter
    def file_name(self, value: str):
        """Set the file name for statistics and solution.
        :param value: The output path.
        :raises ValueError: If the output path is already set."""
        if self.__file_name is not None:
            raise ValueError("Output path is already set.")
        self.__file_name = f'{get_project_root()}/statistic/{value}'

    @property
    def export_solution(self) -> bool:
        """Get whether to show the solution."""
        return self.__export_solution

    @export_solution.setter
    def export_solution(self, value: bool):
        """Set whether to export the solution.
        :param value: True to show the solutions, False otherwise."""
        self.__export_solution = value

    def export(self):
        """Export the statistics to a CSV file and solution to a .sol file."""
        if self.__file_name is None:
            raise ValueError("Output path is not set.")
        self.__statistic.to_csv(self.__file_name, index=False)

        if self.__export_solution:
            solution_file = self.__file_name.replace('.csv', '.sol')
            with open(solution_file, 'w') as f:
                for key, value in self.__solution.items():
                    f.write(f'{key}: {value}\n')

    @property
    def statistic(self) -> DataFrame:
        """Get the DataFrame containing statistics."""
        return self.__statistic

    @property
    def solution(self) -> dict[str, Any]:
        """Get the dictionary containing the solution."""
        return self.__solution
