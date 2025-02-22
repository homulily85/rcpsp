# j = [l for l in range(1, 16)]
# i = [l for l in range(3, 18)]
#
# if len(i) >= len(j):
#     for k in range(i[-1] - i[0] + 2):
#         for t in range(i[0], i[0] + k - 1 + 1):
#             print(f'6_{t}', end=' ')
#         for t in range(j[0] + k, j[-1] + 1):
#             print(f'4_{t}', end=' ')
#         print()
#
# else:
#     for k in range(j[-1] - j[0] + 2):
#         for t in range(j[0], j[0] + k + 1 - 1):
#             print(f'4_{t}', end=' ')
#         for t in range(i[0] + k, i[-1] + 1):
#             print(f'6_{t}', end=' ')
#         print()
from pysat.solvers import Glucose3

from encoder.new_encoder_optimize import Encoder
from encoder.problem import Problem

# p = Problem('data_set/j30.sm/j301_1.sm')
p = Problem('data_set_test/test_2022.sm')

e = Encoder(p, 5)
for j in range(e.problem.njobs):
    print(
        f'Job {j}: ES{j} = {e.ES[j]}, LS{j} = {e.LS[j]}, EC{j} = {e.EC[j]}, LC{j} = {e.LC[j]},'
        f' length = {e.LS[j] - e.ES[j]}')
e.encode()
solver = Glucose3()
for c in e.sat_model.clauses:
    solver.add_clause(c)

print(e.sat_model.number_of_variable)
print(len(e.sat_model.clauses))

t = solver.solve()
print(t)
model = solver.get_model()
a = e.get_result(model)
for i in range(len(a)):
    print(f'Job {i}: {a[i]}')


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
