import os

import networkx as nx
from psplib import parse


class RCPSPProblem:
    """
    Class to represent a problem instance for the RCPSP.
    """

    @staticmethod
    def from_file(file_path: str) -> 'RCPSPProblem':
        """
        Initialize the problem instance by parsing the input file.

        Args:
            file_path (str): Path to the problem instance file.
        Raises:
            ValueError: If the input format is not supported.
        Notes:
            Input file must be either .sm or .rcp format.
        """
        _, file_extension = os.path.splitext(file_path)
        if file_extension not in ['.sm', '.rcp']:
            raise ValueError('Unsupported input format. Supported formats are: .sm, .rcp')
        file_path = os.path.abspath(file_path)

        p = None

        if file_extension == '.sm':
            p = RCPSPProblem.__sm_parse(file_path)
        elif file_extension == '.rcp':
            p = RCPSPProblem.__rcp_parse(file_path)

        return p

    def __init__(self, number_of_activities: int = None,
                 number_of_resources: int = None,
                 durations: list[int] = None,
                 precedence_graph: nx.DiGraph = None,
                 requests: list[list[int]] = None,
                 capacities: list[int] = None):
        """
        Initialize the problem instance.
        Args:
            number_of_activities (int, optional): Number of activities (including dummy start and end activities).
            number_of_resources (int, optional): Number of renewable resources.
            durations (list[int], optional): List of durations for each activity.
            precedence_graph (nx.DiGraph, optional): Directed graph representing the precedence relations between activities.
            requests (list[list[int]], optional): List of resource requests for each activity.
            capacities (list[int], optional): List of capacities for each resource.
        """
        self.__number_of_activities = number_of_activities
        self.__number_of_resources = number_of_resources
        self.__durations = durations
        self.__precedence_graph = precedence_graph if precedence_graph is not None else nx.DiGraph()
        self.__requests = requests
        self.__capacities = capacities
        self.__file_path = None

    @property
    def file_path(self) -> str:
        """
        Return the file path of the problem instance.

        Returns:
            str: The file path of the problem instance.
        """
        return self.__file_path

    @property
    def capacities(self) -> list[int]:
        """
        Return the capacities of the resources.

        Returns:
            list[int]: List of capacities for each resource.
        """
        return self.__capacities

    @property
    def number_of_activities(self) -> int:
        """
        Return the number of activities in the problem instance.

        Returns:
            int: Number of activities (including dummy start and end activities).
        """
        return self.__number_of_activities

    @property
    def number_of_resources(self) -> int:
        """
        Return the number of resources in the problem instance.

        Returns:
            int: Number of renewable resources.
        """
        return self.__number_of_resources

    @property
    def requests(self) -> list[list[int]]:
        """
        Return the resource requests for each activity.

        Returns:
            list[list[int]]: List of resource requests for each activity.
            Each sublist corresponds to an activity and contains the requests for each resource.
        """
        return self.__requests

    @property
    def precedence_graph(self) -> nx.DiGraph:
        """
        Return the precedence graph of the problem instance.

        Returns:
            nx.DiGraph: The directed graph representing the precedence relations between activities.
        """
        return self.__precedence_graph

    @property
    def durations(self) -> list[int]:
        """
        Return the durations of each activity.

        Returns:
            list[int]: List of durations for each activity (including dummy start and end activities).
        """
        return self.__durations

    @staticmethod
    def __sm_parse(file_path) -> 'RCPSPProblem':
        """
        Parse the problem instance from a PSPLIB file.
        """
        instance = parse(file_path, instance_format="psplib")

        problem = RCPSPProblem()

        problem.__file_path = file_path
        # Number of activities (including dummy start and end activities)
        problem.__number_of_activities = instance.num_activities
        # Number of renewable resources
        problem.__number_of_resources = instance.num_resources

        # Duration for each activity
        problem.__durations = [instance.activities[i].modes[0].duration
                               for i in range(problem.__number_of_activities)]

        for i in range(problem.__number_of_activities):
            # Add successors to the precedence graph
            for successors in instance.activities[i].successors:
                problem.__precedence_graph.add_edge(i, successors,
                                                    weight=instance.activities[i].modes[0].duration)

        # Request per resource, per activity
        problem.__requests = [instance.activities[i].modes[0].demands
                              for i in range(problem.__number_of_activities)]
        # Capacity for each per resource
        problem.__capacities = [instance.resources[i].capacity
                                for i in range(problem.__number_of_resources)]

        return problem

    @staticmethod
    def __rcp_parse(file_path) -> 'RCPSPProblem':
        """
        Parse the problem instance from a PACK file.
        """

        problem = RCPSPProblem()
        problem.__file_path = file_path

        with open(file_path, 'r') as f:
            lines = f.readlines()

        # First line: number of jobs and resources
        line_parts = lines[0].strip().split()
        problem.__number_of_activities = int(line_parts[0])
        problem.__number_of_resources = int(line_parts[1])

        # Second line: resource capacities
        problem.__capacities = list(map(int, lines[1].strip().split()))

        # Initialize data structures
        problem.__durations = [0] * problem.__number_of_activities
        problem.__requests = [[] for _ in range(problem.__number_of_activities)]
        successors = [[] for _ in range(problem.__number_of_activities)]
        predecessors = [[] for _ in range(problem.__number_of_activities)]

        # Parse activities
        for i in range(2, len(lines)):
            line_parts = list(map(int, lines[i].strip().split()))
            iterator = iter(line_parts)
            job_idx = i - 2  # Activity index (0-indexed)

            # Duration is the first value
            problem.__durations[job_idx] = next(iterator)

            # Resource requests are the next number_of_resources values
            problem.__requests[job_idx] = [next(iterator) for _ in
                                           range(problem.__number_of_resources)]

            next(iterator)  # Ignore number of successors

            # Successors
            successors[job_idx] = []

            for succ in iterator:
                successors[job_idx].append(succ - 1)

        # Generate predecessors
        for i in range(problem.__number_of_activities):
            for j in successors[i]:
                predecessors[j].append(i)

        # From the pack format, the first activity is made to be immediate predecessor of all other activities.
        # This is not true, so we need to remove it.
        for i in range(problem.__number_of_activities):
            if len(predecessors[i]) > 0 and predecessors[i] != [0]:
                try:
                    predecessors[i].remove(0)
                    successors[0].remove(i)
                except ValueError:
                    pass

        # Create the precedence graph
        for i in range(problem.__number_of_activities):
            for successor in successors[i]:
                problem.__precedence_graph.add_edge(i, successor, weight=problem.__durations[i])

        return problem
