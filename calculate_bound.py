import random
from queue import Queue
import math
import sys
import os
import csv

from encoder.model.problem import Problem

# Constants from the original C++ code
TOURN_FACTOR = 0.5
OMEGA1 = 0.4
OMEGA2 = 0.6


def calcBoundsPriorityRule(problem):
    """
    Tournament heuristic using a priority rule, used for calculating initial lower and upper bounds on the makespan.

    Args:
        problem: problem instance to consider

    Returns:
        tuple of integers (lower_bound, upper_bound)
    """
    horizon = 30000

    # Use a queue for breadth-first traversal of the precedence graph
    q = Queue()
    # Earliest feasible finish time for each job
    ef = [0] * problem.number_of_activities

    q.put(0)
    while not q.empty():
        job = q.get()
        duration = problem.durations[job]
        # Move finish until it is feasible considering resource constraints
        feasibleFinal = False
        while not feasibleFinal:
            feasible = True
            for k in range(problem.number_of_resources):
                # In Python version, requests are not time-dependent
                if problem.requests[job][k] > problem.capacities[k]:
                    feasible = False
                    ef[job] += 1
            if feasible:
                feasibleFinal = True
            if ef[job] > horizon:
                return (0, horizon)

        # Update finish times, and enqueue successors
        for successor in problem.successors[job]:
            f = ef[job] + problem.durations[successor]
            if f > ef[successor]:
                ef[successor] = f  # Use maximum values, because we are interested in critical paths
            q.put(successor)  # Enqueue successor

    # Latest feasible start time for each job
    ls = [horizon] * problem.number_of_activities
    q.put(problem.number_of_activities - 1)

    while not q.empty():
        job = q.get()
        duration = problem.durations[job]
        # Move start until it is feasible considering resource constraints
        feasibleFinal = False
        while not feasibleFinal:
            feasible = True
            for k in range(problem.number_of_resources):
                # In Python version, requests are not time-dependent
                if problem.requests[job][k] > problem.capacities[k]:
                    feasible = False
                    ls[job] -= 1
            if feasible:
                feasibleFinal = True
            if ls[job] < 0:
                return (ef[-1], horizon)

        # Update start times, and enqueue predecessors
        for predecessor in problem.predecessors[job]:
            s = ls[job] - problem.durations[predecessor]
            if s < ls[predecessor]:
                ls[predecessor] = s  # Use minimum values for determining critical paths
            q.put(predecessor)  # Enqueue predecessor

    # Check if any time window is too small: lf[i]-es[i]<durations[i]
    for i in range(problem.number_of_activities):
        if (ls[i] + problem.durations[i]) - (ef[i] - problem.durations[i]) < problem.durations[i]:
            return (ef[-1], horizon)

    # Calculate extended resource utilization values
    ru = [0.0] * problem.number_of_activities
    q.put(problem.number_of_activities - 1)  # Enqueue sink job

    while not q.empty():
        job = q.get()
        duration = problem.durations[job]
        demand = 0
        availability = 0

        for k in range(problem.number_of_resources):
            # In Python version, demand is not time-dependent
            demand += problem.requests[job][k]
            # Simple calculation for availability
            availability += problem.capacities[k] * (ls[job] + duration - (ef[job] - duration))

        ru[job] = OMEGA1 * ((len(problem.successors[job]) / problem.number_of_resources) * (
            demand / availability if availability > 0 else 0))

        for successor in problem.successors[job]:
            ru[job] += OMEGA2 * ru[successor]

        if math.isnan(ru[job]) or ru[job] < 0.0:
            ru[job] = 0.0  # Prevent errors from strange values here

        # Enqueue predecessors
        for predecessor in problem.predecessors[job]:
            q.put(predecessor)

    # Calculate the CPRU (critical path and resource utilization) priority value for each activity
    cpru = [0.0] * problem.number_of_activities
    for job in range(problem.number_of_activities):
        cp = horizon - ls[job]  # Critical path length
        cpru[job] = cp * ru[job]

    # Use fixed seed for deterministic bounds
    random.seed(42)

    # Run a number of passes ('tournaments')
    available = [problem.capacities.copy() for _ in range(horizon)]
    schedule = [-1] * problem.number_of_activities  # Finish time for each process
    schedule[0] = 0  # Schedule the starting dummy activity

    bestMakespan = sys.maxsize // 2

    for _ in range((problem.number_of_activities - 2) * 5):  # Number of passes scales with number of jobs
        for i in range(1, problem.number_of_activities):
            schedule[i] = -1

        # Initialize remaining resource availabilities for all time points
        available = [[problem.capacities[k] for _ in range(horizon)] for k in
                     range(problem.number_of_resources)]

        # Schedule the starting dummy activity
        schedule[0] = 0

        # Schedule all remaining jobs
        for i in range(1, problem.number_of_activities):
            # Randomly select a fraction of the eligible activities (with replacement)
            eligible = []
            for j in range(1, problem.number_of_activities):
                if schedule[j] >= 0:
                    continue

                predecessorsScheduled = True
                for predecessor in problem.predecessors[j]:
                    if schedule[predecessor] < 0:
                        predecessorsScheduled = False

                if predecessorsScheduled:
                    eligible.append(j)

            if not eligible:
                break

            Z = max(int(TOURN_FACTOR * len(eligible)), 2)
            selected = []

            for _ in range(Z):
                choice = int(random.random() * len(eligible))
                selected.append(eligible[choice])

            # Select the activity with the best priority value
            winner = -1
            bestPriority = -sys.float_info.max / 2.0

            for sjob in selected:
                if cpru[sjob] >= bestPriority:
                    bestPriority = cpru[sjob]
                    winner = sjob

            # Schedule it as early as possible
            finish = -1
            for predecessor in problem.predecessors[winner]:
                newFinish = schedule[predecessor] + problem.durations[winner]
                if newFinish > finish:
                    finish = newFinish

            duration = problem.durations[winner]
            feasibleFinal = False

            while not feasibleFinal:
                feasible = True
                for k in range(problem.number_of_resources):
                    for t in range(finish - duration, finish):
                        if t < 0 or t >= horizon:
                            continue
                        # Check if enough resources are available at time t
                        if problem.requests[winner][k] > available[k][t]:
                            feasible = False
                            finish += 1
                            break

                if feasible:
                    feasibleFinal = True
                if finish > horizon:
                    feasibleFinal = False
                    break

            if not feasibleFinal:
                break  # Skip the rest of this pass

            schedule[winner] = finish

            # Update remaining resource availabilities
            for k in range(problem.number_of_resources):
                for t in range(finish - duration, finish):
                    if t < 0 or t >= horizon:
                        continue
                    available[k][t] -= problem.requests[winner][k]

        if schedule[-1] >= 0 and schedule[-1] < bestMakespan:
            bestMakespan = schedule[-1]

    # For the lower bound we use earliest start of end dummy activity
    return (ef[-1], min(horizon, bestMakespan))


def find_file(directory):
    """Find all problem files in the pack_folder directory"""
    problem_files = []

    # Check if directory exists
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist")
        return problem_files

    # Walk through directory
    for root, _, files in os.walk(directory):
        for file in files:
            # Add files that appear to be problem instances
            full_path = os.path.join(root, file)
            problem_files.append(full_path)

    return problem_files


def calculate_and_save_bounds(input_folder, output_file):
    """Calculate bounds for all pack files and save to CSV"""
    # Create bound directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    problem_files = find_file(input_folder)

    if not problem_files:
        print("No problem files found")
        return

    results = []

    # Process each file
    for file_path in problem_files:
        print(f"Processing {file_path}")

        try:
            # Load problem
            problem = Problem(file_path)

            # Calculate bounds
            lower_bound, upper_bound = calcBoundsPriorityRule(problem)

            # Get just the filename (not the full path)
            filename = os.path.basename(file_path)

            # Store result
            results.append((filename, lower_bound, upper_bound))

            print(f"  Bounds: LB={lower_bound}, UB={upper_bound}")

        except Exception as e:
            print(f"  Error processing {file_path}: {e}")

    # Save results to CSV
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        for result in results:
            writer.writerow(result)

    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    calculate_and_save_bounds("data_set/pack", "bound/bound_pack.csv")