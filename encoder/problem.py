from psplib import parse
import os.path


class Problem:
    def __init__(self, input_file: str):
        self.name = input_file
        # Extract dataset name from the path (data_set/{data_set_name}/{problem_name})
        path_parts = input_file.split(os.path.sep)
        self.data_set_name = path_parts[-2] if len(path_parts) >= 2 else ""

        problem_data = self.psplib_parse() if self.data_set_name in ['j30.sm', 'j60.sm', 'j90.sm',
                                                                     'j120.sm'] else self.pack_parse()

        # Set class attributes from parsed data
        self.njobs = problem_data["njobs"]
        self.nresources = problem_data["nresources"]
        self.successors = problem_data["successors"]
        self.predecessors = problem_data["predecessors"]
        self.durations = problem_data["durations"]
        self.requests = problem_data["requests"]
        self.capacities = problem_data["capacities"]

    def psplib_parse(self):
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
            "njobs": njobs,
            "nresources": nresources,
            "successors": successors,
            "predecessors": predecessors,
            "durations": durations,
            "requests": requests,
            "capacities": capacities
        }

    def pack_parse(self):
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

            # Resource requests are the next nresources values
            requests[job_idx] = line_parts[1:1 + nresources]

            # Number of successors
            num_successors = line_parts[1 + nresources]

            # Successors - convert from 1-based to 0-based indexing
            successors[job_idx] = [succ - 1 for succ in line_parts[2 + nresources:2 + nresources + num_successors]]

        # Generate predecessors
        for i in range(njobs):
            for j in successors[i]:
                predecessors[j].append(i)

        return {
            "njobs": njobs,
            "nresources": nresources,
            "successors": successors,
            "predecessors": predecessors,
            "durations": durations,
            "requests": requests,
            "capacities": capacities
        }

