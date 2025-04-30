import csv
import os

from encoder.problem import Problem


def valid_activities_at_time(I, S, R, t):
    """Returns list of valid activities from R that can start at time t, based on precedence and resource constraints."""
    valid = []
    for i in R:
        # Check precedence constraints
        if not all(S[j] + I.durations[j] <= t for j in I.predecessors[i]):
            continue

        # Check resource constraints
        resource_usage = [0 for _ in range(I.number_of_resources)]
        for a in range(len(S)):
            if S[a] <= t < S[a] + I.durations[a] and a not in R:
                for r in range(I.number_of_resources):
                    resource_usage[r] += I.requests[i][r]

        if all(resource_usage[r] + I.requests[i][r] <= I.capacities[r] for r in
               range(I.number_of_resources)):
            valid.append(i)

    return valid


def PSGS(I):
    """Implements Parallel Schedule Generation Scheme (PSGS) for RCPSP instance I."""
    V = list(range(I.number_of_activities))
    R = set(V) - {0}
    A = {0}
    C = set()
    S = [0] * I.number_of_activities

    while R:
        # Compute next event time t
        t = min(S[i] + I.durations[i] for i in A)

        # Activities finishing at time t
        Ct = {i for i in A if S[i] + I.durations[i] == t}
        A -= Ct
        C |= Ct

        # Get valid activities at time t
        D = valid_activities_at_time(I, S, R, t)

        while D:
            i = D.pop()
            S[i] = t
            A.add(i)
            R.remove(i)
            D = valid_activities_at_time(I, S, R, t)

    return [min(S), max(S)]  # Return lower and upper bounds


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
            lower_bound, upper_bound = PSGS(problem)

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
