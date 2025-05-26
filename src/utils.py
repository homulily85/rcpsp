import datetime
import os
from pathlib import Path
from threading import Lock


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