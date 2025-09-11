import logging
import os
import timeit

import networkx as nx
from psplib import parse


class MRCPSPProblem:
    """
    Class to represent a problem instance for the MRCPSP.
    """

    @staticmethod
    def from_file(file_path: str):
        """
        Initialize the problem instance by parsing the input file.

        Args:
            file_path (str): Path to the problem instance file.
        Raises:
            ValueError: If the input format is not supported.
        Notes:
            Input file must be .mm format.
        """

        _, file_extension = os.path.splitext(file_path)
        if file_extension != ".mm":
            logging.critical('Unsupported input format. Supported formats is .mm')
            raise ValueError('Unsupported input format. Supported formats is .mm')
        file_path = os.path.abspath(file_path)
        logging.info(
            f"Parsing the problem instance from {file_path}")
        start = timeit.default_timer()

        p = MRCPSPProblem.__mm_parse(file_path)

        logging.info(
            f"Finished parsing the problem instance from {file_path}")
        logging.info(f"Parsing time: {round(timeit.default_timer() - start, 5)} seconds.")

        return p

    def __init__(self, number_of_activities=0,
                 number_of_renewable_resources=0,
                 number_of_non_renewable_resources=0, durations=None,
                 precedence_graph=None, renewable_resources_request=None,
                 non_renewable_resources_request=None,
                 renewable_resources_capacities=None,
                 non_renewable_resources_capacities=None):
        """
        Initialize the problem instance.
        Args:
            number_of_activities (int): Number of activities in the problem instance.
            number_of_renewable_resources (int): Number of renewable resources in the problem instance.
            number_of_non_renewable_resources (int): Number of non-renewable resources in the problem instance.
            durations (list[list[int]]): A list where each element is a list of durations for each mode of
                the corresponding activity.
            precedence_graph (nx.DiGraph): The directed graph representing the precedence relations between activities.
                Each edge's weight corresponds to the minimum duration of the predecessor activity across all its modes.
            renewable_resources_request (list[list[list[int]]]): A 3D list where each element corresponds to an activity,
                containing a list of modes, each of which contains a list of resource requests for renewable resources.
            non_renewable_resources_request (list[list[list[int]]]): A 3D list where each element corresponds to an activity,
                containing a list of modes, each of which contains a list of resource requests for non-renewable resources.
            renewable_resources_capacities (list[int]): A list where each element corresponds to the capacity of a renewable resource.
            non_renewable_resources_capacities (list[int]): A list where each element corresponds to the capacity of a non-renewable resource.
        """
        self.__file_path = None
        self.__number_of_activities = number_of_activities
        self.__number_of_resources = number_of_renewable_resources + number_of_non_renewable_resources
        self.__number_of_renewable_resources = number_of_renewable_resources
        self.__number_of_non_renewable_resources = number_of_non_renewable_resources
        self.__durations = durations
        self.__precedence_graph = precedence_graph if precedence_graph is not None else nx.DiGraph()
        self.__renewable_resources_request = renewable_resources_request
        self.__non_renewable_resources_request = non_renewable_resources_request
        self.__renewable_resources_capacities = renewable_resources_capacities
        self.__non_renewable_resources_capacities = non_renewable_resources_capacities

    @property
    def number_of_modes(self):
        """
        Returns the number of modes for each activity.
        Returns:
            list[int]: A list where each element corresponds to the number of modes for the
            corresponding activity.
        """
        return [len(modes) for modes in self.__durations]

    @property
    def file_path(self):
        """
        Returns the file path of the problem instance.
        Returns:
            str: The file path of the problem instance.
        """
        return self.__file_path

    @property
    def number_of_activities(self):
        """
        Returns the number of activities in the problem instance.
        Returns:
            int: The number of activities in the problem instance.
        """
        return self.__number_of_activities

    @property
    def number_of_resources(self):
        """
        Returns the number of resources in the problem instance.
        Returns:
            int: The number of resources in the problem instance.
        """
        return self.__number_of_resources

    @property
    def number_of_renewable_resources(self):
        """
        Returns the number of renewable resources in the problem instance.
        Returns:
            int: The number of renewable resources in the problem instance.
        """
        return self.__number_of_renewable_resources

    @property
    def number_of_non_renewable_resources(self):
        """
        Returns the number of non-renewable resources in the problem instance.
        Returns:
            int: The number of non-renewable resources in the problem instance.
        """
        return self.__number_of_non_renewable_resources

    @property
    def durations(self):
        """
        Returns the durations of activities in different modes.
        Returns:
            list[list[int]]: A list where each element is a list of durations for each mode of
            the corresponding activity.
        """
        return self.__durations

    @property
    def precedence_graph(self):
        """
        Returns the precedence graph of the problem instance.
        Returns:
                nx.DiGraph: The directed graph representing the precedence relations between activities.
        Notes:
            Each edge's weight corresponds to the minimum duration of the predecessor activity
            across all its modes.
        """
        return self.__precedence_graph

    @property
    def renewable_resources_request(self):
        """
        Returns the resource requests for renewable resources.
        Returns:
                list[list[list[int]]]: A 3D list where each element corresponds to an activity,
                containing a list of modes, each of which contains a list of resource requests
                for renewable resources.
        """
        return self.__renewable_resources_request

    @property
    def non_renewable_resources_request(self):
        """
        Returns the resource requests for non-renewable resources.
        Returns:
                list[list[list[int]]]: A 3D list where each element corresponds to an activity,
                containing a list of modes, each of which contains a list of resource requests
                for non-renewable resources.
        """
        return self.__non_renewable_resources_request

    @property
    def renewable_resources_capacities(self):
        """
        Returns the capacities of renewable resources.
        Returns:
                list[int]: A list where each element corresponds to the capacity of a renewable resource.
        """
        return self.__renewable_resources_capacities

    @property
    def non_renewable_resources_capacities(self):
        """
        Returns the capacities of non-renewable resources.
        Returns:
                list[int]: A list where each element corresponds to the capacity of a non-renewable resource.
        """
        return self.__non_renewable_resources_capacities

    @staticmethod
    def __mm_parse(file_path):
        """
        Parse the problem instance from a .mm file.
        """
        instance = parse(file_path, instance_format="psplib")

        p = MRCPSPProblem()
        p.__file_path = file_path

        p.__number_of_activities = instance.num_activities
        p.__number_of_resources = instance.num_resources

        for resource in instance.resources:
            if resource.renewable:
                p.__number_of_renewable_resources += 1
            else:
                p.__number_of_non_renewable_resources += 1

        # Duration for each activity in each mode
        p.__durations = [[mode.duration for mode in instance.activities[i].modes]
                         for i in range(p.__number_of_activities)]

        for i in range(p.__number_of_activities):
            # Add successors to the precedence graph
            for successors in instance.activities[i].successors:
                p.__precedence_graph.add_edge(i, successors,
                                              weight=min(
                                                  [instance.activities[i].modes[j].duration for j
                                                   in range(len(instance.activities[i].modes))]))

        # Request per resource, per activity, per mode for renewable resources
        p.__renewable_resources_request = [[[mode.demands[j] for j in
                                             range(p.__number_of_resources)
                                             if instance.resources[j].renewable]
                                            for mode in instance.activities[i].modes]
                                           for i in range(p.__number_of_activities)]
        # Request per resource, per activity, per mode for non-renewable resources
        p.__non_renewable_resources_request = [[[mode.demands[j] for j in
                                                 range(p.__number_of_resources)
                                                 if not instance.resources[j].renewable]
                                                for mode in instance.activities[i].modes]
                                               for i in range(p.__number_of_activities)]
        # Capacity for each per resource for renewable resources
        p.__renewable_resources_capacities = [resource.capacity for resource in
                                              instance.resources
                                              if resource.renewable]
        # Capacity for each per resource for non-renewable resources
        p.__non_renewable_resources_capacities = [resource.capacity for resource in
                                                  instance.resources
                                                  if not resource.renewable]

        return p

    def reduce_non_renewable_resources_request_and_capacity(self):
        for k in range(self.__number_of_non_renewable_resources):
            self.non_renewable_resources_capacities[k] -= sum(min(
                self.non_renewable_resources_request[i][o][k] for o in
                range(self.number_of_modes[i]))
                                                              for i in
                                                              range(1,
                                                                    self.number_of_activities - 1))

        for i in range(1, self.number_of_activities - 1):
            for k in range(self.__number_of_non_renewable_resources):
                min_demand = min(self.non_renewable_resources_request[i][o_prime][k]
                                 for o_prime in range(self.number_of_modes[i]))
                for o in range(self.number_of_modes[i]):
                    self.non_renewable_resources_request[i][o][k] -= min_demand