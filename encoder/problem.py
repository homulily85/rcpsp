import os.path

from psplib import parse


class Problem:
    """
    Class to represent a problem instance for the RCPSP.
    This class is responsible for parsing the problem instance from a file
    and storing the relevant data structures.
    """

    def __init__(self, input_file: str):
        """
        Initialize the problem instance by parsing the input file.
        :param input_file: Path to the input file containing the problem instance.
        :type input_file: str
        """
        self.__name = input_file
        # Extract dataset name from the path (data_set/{dataset_type}/{problem_name})
        path_parts = input_file.split(os.path.sep)
        self.__dataset_type = path_parts[-2] if len(path_parts) >= 2 else ""

        if self.__dataset_type not in ['j30.sm', 'j60.sm', 'j90.sm', 'j120.sm', 'data_set_test',
                                       'pack', 'pack_d']:
            raise ValueError(
                f"Invalid dataset name: {self.__dataset_type}. "
                f"Expected one of "
                f"['j30.sm', 'j60.sm', 'j90.sm', 'j120.sm', 'data_set_test', 'pack','pack_d']")

        problem_data = self.__psplib_parse() \
            if (self.__dataset_type in
                ['j30.sm', 'j60.sm', 'j90.sm', 'j120.sm', 'data_set_test']) \
            else self.__pack_parse()

        # Set class attributes from parsed data
        self.__number_of_activities = problem_data["number_of_activities"]
        self.__number_of_resources = problem_data["number_of_resources"]
        self.__successors = problem_data["successors"]
        self.__predecessors = problem_data["predecessors"]
        self.__durations = problem_data["durations"]
        self.__requests = problem_data["requests"]
        self.__capacities = problem_data["capacities"]

    @property
    def name(self) -> str:
        """
        Get the name of the problem instance.
        :return: Name of the problem instance.
        :rtype: str
        """
        return self.__name

    @property
    def dataset_type(self) -> str:
        """
        Get the dataset type of the problem instance.
        :return: Dataset type of the problem instance.
        :rtype: str
        """
        return self.__dataset_type

    @property
    def capacities(self) -> list[int]:
        """
        Get the capacities of the resources.
        :return: List of capacities for each resource.
        :rtype: list[int]
        """
        return self.__capacities

    @property
    def durations(self) -> list[int]:
        """
        Get the durations of the activities.
        :return: List of durations for each activity.
        :rtype: list[int]
        """
        return self.__durations

    @property
    def number_of_activities(self) -> int:
        """
        Get the number of activities in the problem instance.
        :return: Number of activities.
        :rtype: int
        """
        return self.__number_of_activities

    @property
    def number_of_resources(self) -> int:
        """
        Get the number of resources in the problem instance.
        :return: Number of resources.
        :rtype: int
        """
        return self.__number_of_resources

    @property
    def requests(self) -> list[list[int]]:
        """
        Get the resource requests for each activity.
        :return: List of resource requests for each activity.
        :rtype: list[list[int]]
        """
        return self.__requests

    @property
    def successors(self) -> list[list[int]]:
        """
        Get the successors of each activity.
        :return: List of successors for each activity.
        :rtype: list[list[int]]
        """
        return self.__successors

    @property
    def predecessors(self) -> list[list[int]]:
        """
        Get the predecessors of each activity.
        :return: List of predecessors for each activity.
        :rtype: list[list[int]]
        """
        return self.__predecessors

    def __psplib_parse(self) -> dict[str, list | int]:
        """
        Parse the problem instance from a PSPLIB file.
        :return: Parsed data including number of activities, resources, successors, predecessors, durations, requests, and capacities.
        :rtype: dict[str, list|int]
        """
        instance = parse(self.name, instance_format="psplib")

        # Number of activities (including dummy start and end activities)
        njobs = instance.num_activities
        # Number of renewable resources
        nresources = instance.num_resources
        # List of successors for each activity
        successors = [instance.activities[i].successors for i in range(njobs)]
        # List of predecessors for each activity (for backwards traversal of the precedence graph)
        predecessors = [[] for _ in range(njobs)]
        # Duration for each activity
        durations = [instance.activities[i].modes[0].duration for i in range(njobs)]
        # Request per resource, per activity
        requests = [instance.activities[i].modes[0].demands for i in range(njobs)]
        # Capacity for each per resource
        capacities = [instance.resources[i].capacity for i in range(nresources)]

        # Generate predecessors
        for i in range(njobs):
            for j in successors[i]:
                predecessors[j].append(i)

        return {
            "number_of_activities": njobs,
            "number_of_resources": nresources,
            "successors": successors,
            "predecessors": predecessors,
            "durations": durations,
            "requests": requests,
            "capacities": capacities
        }

    def __pack_parse(self)-> dict[str, list | int]:
        """
        Parse the problem instance from a PACK file.
        :return: Parsed data including number of activities, resources, successors, predecessors, durations, requests, and capacities.
        :rtype: dict[str, list|int]
        """
        with open(self.name, 'r') as f:
            lines = f.readlines()

        # First line: number of jobs and resources
        line_parts = lines[0].strip().split()
        njobs = int(line_parts[0])
        nresources = int(line_parts[1])

        # Second line: resource capacities
        capacities = list(map(int, lines[1].strip().split()))

        # Initialize data structures
        durations = [0] * njobs
        requests = [[] for _ in range(njobs)]
        successors = [[] for _ in range(njobs)]
        predecessors = [[] for _ in range(njobs)]

        # Parse activities
        for i in range(2, 2 + njobs):
            line_parts = list(map(int, lines[i].strip().split()))
            job_idx = i - 2  # Activity index (0-indexed)

            # Duration is the first value
            durations[job_idx] = line_parts[0]

            # Resource requests are the next number_of_resources values
            requests[job_idx] = line_parts[1:1 + nresources]

            # Number of successors
            num_successors = line_parts[1 + nresources]

            # Successors - convert from 1-based to 0-based indexing
            successors[job_idx] = [succ - 1 for succ in
                                   line_parts[2 + nresources:2 + nresources + num_successors]]

        # Generate predecessors
        for i in range(njobs):
            for j in successors[i]:
                predecessors[j].append(i)

        return {
            "number_of_activities": njobs,
            "number_of_resources": nresources,
            "successors": successors,
            "predecessors": predecessors,
            "durations": durations,
            "requests": requests,
            "capacities": capacities
        }
