import logging
import math
import os
import random
import sys
import timeit
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue

import networkx as nx
from matplotlib import pyplot as plt
from networkx.algorithms.dag import transitive_closure_dag
from networkx.algorithms.shortest_paths.dense import floyd_warshall
from psplib import parse

from src.minium_path_cover import minimum_path_cover
from src.utils import SATSolver, SOLVER_STATUS, get_project_root, \
    generate_random_filename


class RCPSPProblem:
    """
    Class to represent a problem instance for the RCPSP.
    """

    def __init__(self, file_path: str):
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
            logging.critical('Unsupported input format. Supported formats are: .sm, .rcp')
            raise ValueError('Unsupported input format. Supported formats are: .sm, .rcp')
        file_path = os.path.abspath(file_path)
        logging.info(
            f"Parsing the problem instance from {file_path}")
        start = timeit.default_timer()

        self.__file_path = file_path
        self.__number_of_activities = None
        self.__number_of_resources = None
        self.__durations = None
        self.__precedence_graph = nx.DiGraph()
        self.__requests = None
        self.__capacities = None

        if file_extension == '.sm':
            self.__sm_parse(file_path)
        elif file_extension == '.rcp':
            self.__rcp_parse(file_path)

        logging.info(
            f"Finished parsing the problem instance from {file_path}")
        logging.info(f"Parsing time: {round(timeit.default_timer() - start, 5)} seconds.")

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

    def __sm_parse(self, file_path):
        """
        Parse the problem instance from a PSPLIB file.
        """
        instance = parse(file_path, instance_format="psplib")

        # Number of activities (including dummy start and end activities)
        self.__number_of_activities = instance.num_activities
        # Number of renewable resources
        self.__number_of_resources = instance.num_resources

        # Duration for each activity
        self.__durations = [instance.activities[i].modes[0].duration
                            for i in range(self.__number_of_activities)]

        for i in range(self.__number_of_activities):
            # Add successors to the precedence graph
            for successors in instance.activities[i].successors:
                self.__precedence_graph.add_edge(i, successors,
                                                 weight=instance.activities[i].modes[0].duration)

        # Request per resource, per activity
        self.__requests = [instance.activities[i].modes[0].demands
                           for i in range(self.__number_of_activities)]
        # Capacity for each per resource
        self.__capacities = [instance.resources[i].capacity
                             for i in range(self.__number_of_resources)]

    def __rcp_parse(self, file_path):
        """
        Parse the problem instance from a PACK file.
        """
        with open(file_path, 'r') as f:
            lines = f.readlines()

        # First line: number of jobs and resources
        line_parts = lines[0].strip().split()
        self.__number_of_activities = int(line_parts[0])
        self.__number_of_resources = int(line_parts[1])

        # Second line: resource capacities
        self.__capacities = list(map(int, lines[1].strip().split()))

        # Initialize data structures
        self.__durations = [0] * self.__number_of_activities
        self.__requests = [[] for _ in range(self.__number_of_activities)]
        successors = [[] for _ in range(self.__number_of_activities)]
        predecessors = [[] for _ in range(self.__number_of_activities)]

        # Parse activities
        for i in range(2, len(lines)):
            line_parts = list(map(int, lines[i].strip().split()))
            iterator = iter(line_parts)
            job_idx = i - 2  # Activity index (0-indexed)

            # Duration is the first value
            self.__durations[job_idx] = next(iterator)

            # Resource requests are the next number_of_resources values
            self.__requests[job_idx] = [next(iterator) for _ in range(self.__number_of_resources)]

            next(iterator)  # Ignore number of successors

            # Successors
            successors[job_idx] = []

            for succ in iterator:
                successors[job_idx].append(succ - 1)

        # Generate predecessors
        for i in range(self.__number_of_activities):
            for j in successors[i]:
                predecessors[j].append(i)

        # From the pack format, the first activity is made to be immediate predecessor of all other activities.
        # This is not true, so we need to remove it.
        for i in range(self.__number_of_activities):
            if len(predecessors[i]) > 0 and predecessors[i] != [0]:
                try:
                    predecessors[i].remove(0)
                    successors[0].remove(i)
                except ValueError:
                    pass

        # Create the precedence graph
        for i in range(self.__number_of_activities):
            for successor in successors[i]:
                self.__precedence_graph.add_edge(i, successor, weight=self.__durations[i])


class RCPSPSolver:
    """
    Class to solve the Resource-Constrained Project Scheduling Problem (RCPSP) using SAT solver.
    """

    def __init__(self, problem: RCPSPProblem, lower_bound: int = None, upper_bound: int = None):
        """
        Initialize the RCPSP solver with a problem instance.
        Args:
            problem (RCPSPProblem): The problem instance to solve.
            lower_bound (int, optional): The lower bound for the makespan. If None, it will be calculated.
            upper_bound (int, optional): The upper bound for the makespan. If None, it will be calculated.
        Raises:
            ValueError: If the method is not 'sat' or 'maxsat', or if the bounds are invalid.

        """
        self.__makespan_var = None
        self.__register = {}

        if not isinstance(problem, RCPSPProblem):
            logging.critical("The problem must be an instance of RCPSPProblem.")
            raise ValueError("The problem must be an instance of RCPSPProblem.")

        self.__problem = problem

        if lower_bound is not None and upper_bound is not None:
            if lower_bound < 0 or upper_bound < 0:
                logging.critical("Lower and upper bounds must be non-negative integers.")
                raise ValueError("Lower and upper bounds must be non-negative integers.")

            if lower_bound > upper_bound:
                logging.critical("Lower bound cannot be greater than upper bound.")
                raise ValueError("Lower bound cannot be greater than upper bound.")

        if lower_bound is None or upper_bound is None:
            t = self.__calculate_bound()
            if lower_bound is None:
                lower_bound = t[0]
            if upper_bound is None or upper_bound < lower_bound:
                upper_bound = t[1]

        self.__lower_bound = lower_bound
        self.__upper_bound = upper_bound
        self.__makespan = self.__upper_bound
        self.__failed_preprocessing = False
        self.__schedule = None
        self.__solver = SATSolver()
        self.__status = None
        self.__encoding_time = 0
        self.__preprocessing_time = 0
        self.__preprocessing()

    def __calculate_bound(self):
        logging.info(
            'Calculating lower and upper bounds using the CPRU method...')
        start = timeit.default_timer()
        horizon = 30000
        tourn_factor = 0.5
        omega1 = 0.4
        omega2 = 0.6

        # Use a queue for breadth-first traversal of the precedence graph
        q = Queue()
        # Earliest feasible finish time for each job
        ef = [0] * self.__problem.number_of_activities

        q.put(0)
        while not q.empty():
            job = q.get()
            # Move finish until it is feasible considering resource constraints
            feasible_final = False
            while not feasible_final:
                feasible = True
                for k in range(self.__problem.number_of_resources):
                    # In Python version, requests are not time-dependent
                    if self.__problem.requests[job][k] > self.__problem.capacities[k]:
                        feasible = False
                        ef[job] += 1
                if feasible:
                    feasible_final = True
                if ef[job] > horizon:
                    return 0, horizon

            # Update finish times, and enqueue successors
            for successor in self.__problem.precedence_graph.successors(job):
                f = ef[job] + self.__problem.durations[successor]
                if f > ef[successor]:
                    # Use maximum values, because we are interested in critical paths
                    ef[successor] = f
                q.put(successor)  # Enqueue successor

        # Latest feasible start time for each job
        ls = [horizon] * self.__problem.number_of_activities
        q.put(self.__problem.number_of_activities - 1)

        while not q.empty():
            job = q.get()
            # Move start until it is feasible considering resource constraints
            feasible_final = False
            while not feasible_final:
                feasible = True
                for k in range(self.__problem.number_of_resources):
                    # In Python version, requests are not time-dependent
                    if self.__problem.requests[job][k] > self.__problem.capacities[k]:
                        feasible = False
                        ls[job] -= 1
                if feasible:
                    feasible_final = True
                if ls[job] < 0:
                    return ef[-1], horizon

            # Update start times, and enqueue predecessors
            for predecessor in self.__problem.precedence_graph.predecessors(job):
                s = ls[job] - self.__problem.durations[predecessor]
                if s < ls[predecessor]:
                    ls[predecessor] = s  # Use minimum values for determining critical paths
                q.put(predecessor)  # Enqueue predecessor

        # Check if any time window is too small: lf[i]-es[i]<durations[i]
        for i in range(self.__problem.number_of_activities):
            if (ls[i] + self.__problem.durations[i]) - (ef[i] - self.__problem.durations[i]) < \
                    self.__problem.durations[i]:
                return ef[-1], horizon

        # Calculate extended resource utilization values
        ru = [0.0] * self.__problem.number_of_activities
        q.put(self.__problem.number_of_activities - 1)  # Enqueue sink job

        def get_iterator_size(iterator):
            return sum(1 for _ in iterator)

        while not q.empty():
            job = q.get()
            duration = self.__problem.durations[job]
            demand = 0
            availability = 0

            for k in range(self.__problem.number_of_resources):
                demand += self.__problem.requests[job][k]
                # Simple calculation for availability
                availability += self.__problem.capacities[k] * (
                        ls[job] + duration - (ef[job] - duration))

            ru[job] = omega1 * ((get_iterator_size(self.__problem.precedence_graph.successors(
                job)) / self.__problem.number_of_resources) * (
                                    demand / availability if availability > 0 else 0))

            for successor in self.__problem.precedence_graph.successors(job):
                ru[job] += omega2 * ru[successor]

            if math.isnan(ru[job]) or ru[job] < 0.0:
                ru[job] = 0.0  # Prevent errors from strange values here

            # Enqueue predecessors
            for predecessor in self.__problem.precedence_graph.predecessors(job):
                q.put(predecessor)

        # Calculate the CPRU (critical path and resource utilization) priority value for each activity
        cpru = [0.0] * self.__problem.number_of_activities
        for job in range(self.__problem.number_of_activities):
            cp = horizon - ls[job]  # Critical path length
            cpru[job] = cp * ru[job]

        # Use fixed seed for deterministic bounds
        random.seed(42)

        # Run a number of passes ('tournaments')
        [self.__problem.capacities.copy() for _ in range(horizon)]
        schedule = [-1] * self.__problem.number_of_activities  # Finish time for each process
        schedule[0] = 0  # Schedule the starting dummy activity

        best_makespan = sys.maxsize // 2

        # Number of passes scales with number of jobs
        for _ in range((self.__problem.number_of_activities - 2) * 5):
            for i in range(1, self.__problem.number_of_activities):
                schedule[i] = -1

            # Initialize remaining resource availabilities for all-time points
            available = [[self.__problem.capacities[k] for _ in range(horizon)] for k in
                         range(self.__problem.number_of_resources)]

            # Schedule the starting dummy activity
            schedule[0] = 0

            # Schedule all remaining jobs
            for i in range(1, self.__problem.number_of_activities):
                # Randomly select a fraction of the eligible activities (with replacement)
                eligible = []
                for j in range(1, self.__problem.number_of_activities):
                    if schedule[j] >= 0:
                        continue

                    predecessors_scheduled = True
                    for predecessor in self.__problem.precedence_graph.predecessors(j):
                        if schedule[predecessor] < 0:
                            predecessors_scheduled = False

                    if predecessors_scheduled:
                        eligible.append(j)

                if not eligible:
                    break

                z = max(int(tourn_factor * len(eligible)), 2)
                selected = []

                for _ in range(z):
                    choice = int(random.random() * len(eligible))
                    selected.append(eligible[choice])

                # Select the activity with the best priority value
                winner = -1
                best_priority = -sys.float_info.max / 2.0

                for sjob in selected:
                    if cpru[sjob] >= best_priority:
                        best_priority = cpru[sjob]
                        winner = sjob

                # Schedule it as early as possible
                finish = -1
                for predecessor in self.__problem.precedence_graph.predecessors(winner):
                    new_finish = schedule[predecessor] + self.__problem.durations[winner]
                    if new_finish > finish:
                        finish = new_finish

                duration = self.__problem.durations[winner]
                feasible_final = False

                while not feasible_final:
                    feasible = True
                    for k in range(self.__problem.number_of_resources):
                        for t in range(finish - duration, finish):
                            if t < 0 or t >= horizon:
                                continue
                            # Check if enough resources are available at time t
                            if self.__problem.requests[winner][k] > available[k][t]:
                                feasible = False
                                finish += 1
                                break

                    if feasible:
                        feasible_final = True
                    if finish > horizon:
                        feasible_final = False
                        break

                if not feasible_final:
                    break  # Skip the rest of this pass

                schedule[winner] = finish

                # Update remaining resource availabilities
                for k in range(self.__problem.number_of_resources):
                    for t in range(finish - duration, finish):
                        if t < 0 or t >= horizon:
                            continue
                        available[k][t] -= self.__problem.requests[winner][k]

            if 0 <= schedule[-1] < best_makespan:
                best_makespan = schedule[-1]

        logging.info(
            f"Finished calculating lower and upper bounds in {round(timeit.default_timer() - start, 5)} seconds."
        )
        # For the lower bound we use the earliest start of end dummy activity
        return ef[-1], min(horizon, best_makespan)

    def __preprocessing(self):
        logging.info('Preprocessing started...')
        logging.info('Creating the extended precedence graph...')
        start = timeit.default_timer()
        self.__extended_precedence_graph = transitive_closure_dag(self.__problem.precedence_graph)

        fw = floyd_warshall(self.__problem.precedence_graph)
        results = {a: dict(b) for a, b in fw.items()}

        for source, targets in results.items():
            for target, weight in targets.items():
                if weight != float('inf') and source != target:
                    self.__extended_precedence_graph[source][target]['weight'] = weight

        logging.info(
            f"Finished creating the extended precedence graph in {round(timeit.default_timer() - start, 5)} seconds.")

        logging.info('Updating time lags using energetic reasoning on precedences...')
        start = timeit.default_timer()
        for edge in self.__extended_precedence_graph.edges():
            i, j = edge
            self.__extended_precedence_graph[i][j]['weight'] = max(
                self.__extended_precedence_graph[i][j]['weight'],
                self.__problem.durations[i] + max([self.__rlb(i, j, k) for k in
                                                   range(self.__problem.number_of_resources)]))

        logging.info(
            f"Finished updating time lags in {round(timeit.default_timer() - start, 5)} seconds.")

        logging.info(
            'Calculating ES, EC, LS, and LC for each activity and creating necessary variables...')
        start = timeit.default_timer()
        self.__ES = [self.__extended_precedence_graph[0][i]['weight'] for i in
                     range(1, self.__problem.number_of_activities)]
        self.__ES.insert(0, 0)
        self.__EC = [self.__ES[i] + self.__problem.durations[i] for i in
                     range(self.__problem.number_of_activities)]
        self.__LS = [self.__upper_bound -
                     self.__extended_precedence_graph[i][self.__problem.number_of_activities - 1]
                     ['weight']
                     for i in range(0, self.__problem.number_of_activities - 1)]
        self.__LS.append(self.__upper_bound)  # Last activity ends at upper bound
        self.__LC = [self.__LS[i] + self.__problem.durations[i] for i in
                     range(self.__problem.number_of_activities)]

        if not all(
                self.__ES[i] <= self.__LS[i] for i in range(self.__problem.number_of_activities)):
            logging.info(
                "Preprocessing failed with given upper bound. "
            )
            self.__failed_preprocessing = True
            return

        self.__start = {}
        self.__run = {}

        for i in range(self.__problem.number_of_activities):
            for t in range(self.__ES[i],
                           self.__LS[i] + 1):  # t in STW(i) (start time window of activity i)
                self.__start[i, t] = self.__solver.create_new_variable()

        for i in range(self.__problem.number_of_activities):
            for t in range(self.__ES[i],
                           self.__LC[i]):  # t in RTW(i) (run time window of activity i)
                self.__run[i, t] = self.__solver.create_new_variable()

        self.__preprocessing_time = round(timeit.default_timer() - start, 5)
        logging.info(
            f"Finished calculating ES, EC, LS, and LC in {round(timeit.default_timer() - start, 5)} seconds.")
        logging.info('Preprocessing finished.')

    def __rlb(self, i, j, k) -> int:
        temp = 0
        for a in range(1, self.__problem.number_of_activities - 1):
            if (self.__extended_precedence_graph.has_edge(i, a) and
                    self.__extended_precedence_graph.has_edge(a, j)):
                temp += self.__problem.durations[a] * self.__problem.requests[a][k]

        return math.ceil(1 / self.__problem.capacities[k] * temp)

    def encode(self):
        """
        Encode the problem instance into the solver's format.

        """
        logging.info('Encoding the problem instance...')
        if self.__failed_preprocessing:
            return

        start = timeit.default_timer()
        self.__start_time_for_first_activity()
        self.__start_time_constraint()
        self.__precedence_constraint()
        self.__resource_constraints_with_pbamo()
        self.__consistency_constraint()
        self.__backpropagate_constraint()
        self.__encoding_time = round(timeit.default_timer() - start, 5)

        logging.info(f"Encoding finished in {self.__encoding_time} seconds.")

    def solve(self, time_limit=None, find_optimal: bool = False) -> SOLVER_STATUS:
        """
        Solve the encoded problem instance.
        Args:
            time_limit (int, optional): Time limit for the solver in seconds. If None, no time limit is set.
            find_optimal (bool, optional): If True, find the optimal solution. Defaults to False.

        Returns:
            SOLVER_STATUS: The status of the solver after attempting to solve the problem.
            Possible values are SATISFIABLE, UNSATISFIABLE, OPTIMAL, or UNKNOWN.

        """
        if self.__failed_preprocessing:
            self.__status = SOLVER_STATUS.UNSATISFIABLE
            return self.__status

        if not find_optimal:
            results = self.__solver.solve(time_limit)
            if results is None or results == SOLVER_STATUS.UNKNOWN:
                self.__status = SOLVER_STATUS.UNKNOWN
                return self.__status
            else:
                self.__status = SOLVER_STATUS.SATISFIABLE if results else SOLVER_STATUS.UNSATISFIABLE
                return self.__status
        else:
            result = self.__solver.solve(time_limit)
            if result is None:
                self.__status = SOLVER_STATUS.UNKNOWN
                return self.__status
            if not result:
                self.__status = SOLVER_STATUS.UNSATISFIABLE
                return self.__status

            while (result is not None and result
                   and self.__makespan > self.__lower_bound
                   and self.__makespan > self.__ES[-1]):
                self.__solver.add_assumption(
                    -self.__start[self.__problem.number_of_activities - 1, self.__makespan])
                self.__makespan -= 1
                if time_limit is None:
                    result = self.__solver.solve()
                else:
                    try:
                        result = self.__solver.solve(
                            time_limit - self.__solver.get_statistics()['total_solving_time'])
                    except ValueError:
                        pass

            self.__status = SOLVER_STATUS.SATISFIABLE if result is None else SOLVER_STATUS.OPTIMAL
            return self.__status

    def get_schedule(self) -> list[int] | None:
        """
        Retrieve the schedule after solving the problem instance.

        Returns:
            list[int] | None: The schedule as a list of start times for each activity,
            or None if the problem is unsatisfiable or unknown.
        """
        if self.__status in [SOLVER_STATUS.UNSATISFIABLE, SOLVER_STATUS.UNKNOWN]:
            return None

        if self.__schedule is None:
            logging.info("Retrieving the schedule...")
            start = timeit.default_timer()

            start_times = [-1 for _ in range(self.__problem.number_of_activities)]

            def get_start_time(activity: int) -> int:
                for s in range(self.__ES[activity], self.__LS[activity] + 1):
                    if self.__solver.get_last_feasible_model()[
                        self.__start[(activity, s)] - 1] > 0:
                        return s
                raise Exception(
                    f"Start time for activity {activity} not found in the solution.")

            # Set your desired max number of threads
            max_threads = os.cpu_count()

            # Function wrapper if needed
            def get_start_time_wrapper(i):
                return i, get_start_time(i)

            with ThreadPoolExecutor(max_workers=max_threads) as executor:
                # Submit all tasks
                futures = {executor.submit(get_start_time_wrapper, i): i for i in
                           range(self.__problem.number_of_activities)}

                # Collect results as they complete
                for future in as_completed(futures):
                    i, start_time = future.result()
                    start_times[i] = start_time

            self.__schedule = start_times

            logging.info(
                f"Schedule retrieved successfully in {round(timeit.default_timer() - start, 5)} seconds.")

        return self.__schedule

    def get_makespan(self) -> int | None:
        """
        Retrieve the makespan after solving the problem instance.

        Returns:
            int | None: The makespan of the schedule, or None if the problem is unsatisfiable or unknown.
        """
        if self.__status in [SOLVER_STATUS.UNSATISFIABLE, SOLVER_STATUS.UNKNOWN]:
            return None

        return self.get_schedule()[-1]

    def get_graph(self, save_to_a_file=False, width=None, height=None):
        """
        Generate and display or save a tree-like graph of the schedule.
        Args:
            save_to_a_file: If True, save the graph to a file instead of displaying it.
            width: Width of the graph figure.
            height: Height of the graph figure.
        """
        logging.info(f'Building the schedule graph...')
        start = timeit.default_timer()

        g = self.__problem.precedence_graph.copy()
        schedule = self.get_schedule()

        # ---- Dynamically determine figure size ----
        try:
            levels = nx.algorithms.dag_longest_path_length(g) + 1
        except nx.NetworkXUnfeasible:
            levels = int(len(g.nodes) ** 0.5)  # fallback if not a DAG

        # Estimate width as max branching factor
        level_count = {}
        for node in nx.topological_sort(g):
            depth = len(nx.ancestors(g, node))
            level_count[depth] = level_count.get(depth, 0) + 1
        max_width = max(level_count.values()) if level_count else 1

        # Scale figure size (each node ~2.5x2.5 inches)
        fig_w = (width if width else max(10, max_width * 2.5))
        fig_h = (height if height else max(8, levels * 2.5))
        plt.figure(figsize=(fig_w, fig_h))

        # ---- Layout (tree-like with spacing tweaks) ----
        pos = nx.nx_agraph.graphviz_layout(
            g, prog="dot", args="-Granksep=2.0 -Gnodesep=0.25"
        )

        # Draw nodes
        node_size = 1200
        nx.draw_networkx_nodes(
            g, pos, node_size=node_size, node_color="lightblue", edgecolors="black"
        )
        nx.draw_networkx_edges(g, pos, arrows=True, arrowstyle="->", arrowsize=20)

        # Node labels = index
        nx.draw_networkx_labels(
            g, pos, labels={i: str(i) for i in g.nodes()}, font_weight="bold"
        )

        # Edge labels (if any)
        edge_labels = nx.get_edge_attributes(g, "weight")
        if edge_labels:
            nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels, font_color="red")

        # ---- Schedule labels below nodes (dynamic offset) ----
        # offset proportional to figure height & node size
        y_range = max(y for _, y in pos.values()) - min(y for _, y in pos.values())
        offset = max(-0.025 * y_range, -15)  # 5% of graph height
        schedule_labels = {i: f"S({i}) = {schedule[i]}" for i in g.nodes()}
        offset_pos = {node: (x, y + offset) for node, (x, y) in pos.items()}
        nx.draw_networkx_labels(g, offset_pos, labels=schedule_labels, font_color="blue")

        if save_to_a_file:
            os.makedirs(os.path.join(get_project_root(), 'graphs'), exist_ok=True)
            output_path = os.path.join(
                get_project_root(), 'graphs', f'{generate_random_filename()}.png'
            )
            plt.savefig(output_path, bbox_inches="tight")
            logging.info(f'Schedule graph saved to {output_path}')
        else:
            plt.show()

        plt.close()
        logging.info(
            f'Schedule graph built successfully in {round(timeit.default_timer() - start, 5)} seconds.'
        )

    def __start_time_for_first_activity(self):
        self.__solver.add_clause([self.__start[(0, 0)]])

    def __resource_constraints_with_pbamo(self):
        pb_clauses = []
        for t in range(self.__upper_bound):
            for r in range(self.__problem.number_of_resources):
                literals = []
                weights = []
                for i in range(1, self.__problem.number_of_activities):
                    if t in range(self.__ES[i], self.__LC[i]) and self.__problem.requests[i][r] > 0:
                        if self.__problem.requests[i][r] > 0:
                            literals.append(self.__run[i, t])
                            weights.append(self.__problem.requests[i][r])

                if sum(weights) > self.__problem.capacities[r]:
                    pb_clauses.append((literals, weights, self.__problem.capacities[r]))

            g = nx.DiGraph()
            index_to_label = {}
            label_to_index = {}
            nodes = []
            count = 0
            for i in range(self.__problem.number_of_activities):
                if t in range(self.__ES[i], self.__LC[i]):
                    index_to_label[count] = i
                    label_to_index[i] = count
                    nodes.append(count)
                    count += 1
            g.add_nodes_from(nodes)

            edges = []
            for e in self.__extended_precedence_graph.edges:
                if e[0] in label_to_index and e[1] in label_to_index:
                    edges.append([label_to_index[e[0]], label_to_index[e[1]]])
            g.add_edges_from(edges)

            path_cover = minimum_path_cover(g.number_of_nodes(), g.edges)

            og_path_cover = []

            for path in path_cover:
                og_path_cover.append([index_to_label[i] for i in path])

            for p in og_path_cover:
                pb_clauses.append(([self.__run[i, t] for i in p],
                                   [1 for _ in p], 1))

        self.__solver.add_pb_clauses(pb_clauses)

    def __consistency_constraint(self):
        for i in range(self.__problem.number_of_activities):
            for s in range(self.__ES[i], self.__LS[i] + 1):
                for t in range(s, s + self.__problem.durations[i]):
                    self.__solver.add_clause([-self.__start[i, s], self.__run[i, t]])

    def __backpropagate_constraint(self):
        for i in range(self.__problem.number_of_activities):
            for t in range(self.__EC[i], self.__LC[i] - 1):
                self.__solver.add_clause([-self.__run[i, t], self.__run[i, t + 1],
                                          self.__start[i, t - self.__problem.durations[i] + 1]])

    def __start_time_constraint(self):
        for i in range(1, self.__problem.number_of_activities):
            self.__solver.add_clause(
                [self.__get_forward_staircase_register(i, self.__ES[i], self.__LS[i] + 1)])

    def __precedence_constraint(self):
        for predecessor in range(1, self.__problem.number_of_activities):
            for successor in self.__extended_precedence_graph.successors(predecessor):
                # Precedence constraint
                for k in range(self.__ES[successor], self.__LS[successor] + 1):
                    if k - self.__extended_precedence_graph[predecessor][successor].get(
                            "weight") + 1 >= self.__LS[predecessor] + 1:
                        continue

                    first_half = self.__get_forward_staircase_register(successor,
                                                                       self.__ES[successor],
                                                                       k + 1)

                    t = k - self.__extended_precedence_graph[predecessor][successor].get(
                        "weight") + 1
                    if t < self.__ES[predecessor]:
                        t = self.__ES[predecessor]

                    self.__solver.add_clause([-first_half, -self.__start[predecessor, t]])

                for k in range(
                        self.__LS[successor] - self.__extended_precedence_graph[predecessor][
                            successor].get("weight") + 2,
                        self.__LS[predecessor] + 1):
                    self.__solver.add_clause(
                        [-self.__get_forward_staircase_register(successor,
                                                                self.__ES[successor],
                                                                self.__LS[successor] + 1),
                         -self.__start[predecessor, k]])

    def __get_forward_staircase_register(self, job: int, start: int, end: int) -> int | None:
        """Get forward staircase register for a job for a range of time [start, end)"""
        # Check if current job with provided start and end time is already in the register
        if start >= end:
            return None
        temp = tuple(self.__start[(job, s)] for s in range(start, end))
        if temp in self.__register:
            return self.__register[temp]

        # Store the start time of the job
        accumulative = []
        for s in range(start, end):
            accumulative.append(self.__start[(job, s)])
            current_tuple = tuple(accumulative)
            # If the current tuple is not in the register, create a new variable
            if current_tuple not in self.__register:
                # Create a new variable for the current tuple
                self.__register[current_tuple] = self.__solver.create_new_variable()

                # Create constraint for staircase

                # If current tuple is true then the register associated with it must be true
                self.__solver.add_clause(
                    [-self.__start[(job, s)], self.__register[current_tuple]])

                if s == start:
                    self.__solver.add_clause(
                        [self.__start[(job, s)], -self.__register[current_tuple]])
                else:
                    # Get the previous tuple
                    previous_tuple = tuple(accumulative[:-1])
                    # If previous tuple is true then the current register must be true
                    self.__solver.add_clause(
                        [-self.__register[previous_tuple], self.__register[current_tuple]])

                    # Both previous tuple and current variable is false then current tuple must be false
                    self.__solver.add_clause(
                        [self.__register[previous_tuple], self.__start[job, s],
                         -self.__register[current_tuple]])

                    # Previous tuple and current variable must not be true at the same time
                    self.__solver.add_clause(
                        [-self.__register[previous_tuple], -self.__start[(job, s)]])

        return self.__register[temp]

    def verify(self):
        """
        Verify the solution of the problem.
        This method checks if the solution satisfies all constraints, including precedence and resource constraints.
        If any constraint is violated, it logs an error and exits the program.
        """
        # Get start time
        solution = self.get_schedule()

        if solution is None:
            return

        # Check precedence constraint
        logging.info("Verifying the solution...")
        start = timeit.default_timer()
        for job in range(self.__problem.number_of_activities):
            for predecessor in self.__problem.precedence_graph.predecessors(job):
                if solution[job] < solution[predecessor] + self.__problem.durations[predecessor]:
                    logging.error(
                        f"Failed when checking precedence constraint for {predecessor} -> {job}")
                    print(f"Failed when checking precedence constraint for {predecessor} -> {job}")
                    exit(-1)

        # Checking resource constraint
        for t in range(solution[-1] + 1):
            for r in range(self.__problem.number_of_resources):
                total_consume = 0
                for j in range(self.__problem.number_of_activities):
                    if solution[j] <= t <= solution[j] + self.__problem.durations[j] - 1:
                        total_consume += self.__problem.requests[j][r]
                if total_consume > self.__problem.capacities[r]:
                    logging.error(
                        f"Failed when check resource constraint for resource {r} at t = {t}")
                    print(f"Failed when check resource constraint for resource {r} at t = {t}")
                    exit(-1)

        logging.info("Solution verified successfully. All constraints are satisfied.")
        logging.info(f"Verification time: {round(timeit.default_timer() - start, 5)} seconds.")

    def get_statistics(self) -> dict[str, int | float]:
        """
        Get the statistics of the solver.

        Returns:
            dict[str, int | float]: A dictionary containing various statistics about the solving process.

            This dictionary includes:
                - file_path: The path to the problem file.
                - lower_bound: The lower bound of the problem.
                - upper_bound: The upper bound of the problem.
                - variables: The number of variables used in the solver.
                - clauses: The number of clauses used in the solver.
                - status: The status of the solver (e.g., SATISFIABLE, UNSATISFIABLE, OPTIMAL, UNKNOWN).
                - makespan: The makespan of the schedule, if available.
                - preprocessing_time: The time taken for preprocessing the problem.
                - encoding_time: The time taken for encoding the problem.
                - total_solving_time: The total time taken to solve the problem.
        """
        t = self.__solver.get_statistics()
        return {
            'file_path': self.__problem.file_path,
            'lower_bound': self.__lower_bound,
            'upper_bound': self.__upper_bound,
            'variables': t['variables'],
            'clauses': t['clauses'],
            'status': self.__status.name,
            'makespan': self.get_makespan(),
            'preprocessing_time': self.__preprocessing_time,
            'encoding_time': self.__encoding_time,
            'total_solving_time': t['total_solving_time'],
        }

    def clear_interrupt(self):
        """
        Clear any interrupt set on the solver.
        This method is used to reset the solver's state if it has been interrupted by a timeout.
        """
        self.__solver.clear_interrupt()
