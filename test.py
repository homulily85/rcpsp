import timeit

from pysat.solvers import Glucose42

from encoder.staircase import Encoder
from encoder.problem import Problem

p = Problem('data_set/j30.sm/j305_6.sm')
# p = Problem('data_set_test/test_2022.sm')

e = Encoder(p, 64)
for j in range(e.problem.njobs):
    print(
        f'Job {j}: ES{j} = {e.ES[j]}, LS{j} = {e.LS[j]}, EC{j} = {e.EC[j]}, LC{j} = {e.LC[j]},'
        f' length = {e.LS[j] - e.ES[j]}')

e.encode()
solver = Glucose42()
for c in e.sat_model.clauses:
    solver.add_clause(c)

# print('Number of vars: ', e.sat_model.number_of_variable)
# print('Number of clauses: ', len(e.sat_model.clauses))

start_time = timeit.default_timer()
t = solver.solve()
solve_time = timeit.default_timer() - start_time
print(solve_time)

model = solver.get_model()

# a = e.get_result(model)
# for i in range(len(a)):
#     print(f'Job {i}: {a[i]}')


def verify(encoder: Encoder, model: list[int]):
    # Get start time
    start_time = encoder.get_result(model)

    # Check precedence constraint
    for activity in range(encoder.problem.njobs):
        for successor in encoder.problem.successors[activity]:
            if start_time[successor] < start_time[activity] + encoder.problem.durations[activity]:
                print(f"Failed when checking precedence constraint for {activity} ->{successor}")
                exit(-1)
    # Checking resource constraint
    for t in range(start_time[-1] + 1):
        total_request = [0 for _ in range(encoder.problem.nresources)]
        for activity in range(encoder.problem.njobs):
            if start_time[activity] <= t <= start_time[activity] + e.problem.durations[
                activity] - 1:
                for r in range(e.problem.njobs):
                    total_request += e.problem.requests[r]

        for r in range(encoder.problem.nresources):
            if total_request[r] > encoder.problem.capacities[r]:
                print(f"Failed when check resource constraint for resource {r} at t = {t}")
                exit(-1)


verify(e, model)
