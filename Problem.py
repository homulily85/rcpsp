from psplib import parse


class Problem:
    def __init__(self, input_file: str):
        instance = parse(input_file, instance_format="psplib")
        # Number of activities (including dummy start and end activities)
        self.njobs = instance.num_activities
        # Number of renewable resources
        self.nresources = instance.num_resources
        # List of successors for each activity
        self.successors = [instance.activities[i].successors for i in range(self.njobs)]
        # List of predecessors for each activity (for backwards traversal of the precedence graph)
        self.predecessors = [[] for _ in range(self.njobs)]
        # Duration for each activity
        self.durations = [instance.activities[i].modes[0].duration for i in range(self.njobs)]
        # Request per time step, per resource, per activity
        self.requests = [
            [[instance.activities[job_no].modes[0].demands[res_no] for _ in
              range(self.durations[job_no])] for res_no in
             range(self.nresources)] for job_no in range(self.njobs)]
        # Capacity for each time step, per resource
        self.capacities = [[instance.resources[res_no].capacity for _ in range(10)] for res_no in
                           range(instance.num_resources)]

        for i in range(self.njobs):
            for j in self.successors[i]:
                self.predecessors[j].append(i)


p = Problem('j301_1.sm')

print(p)