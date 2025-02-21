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

from encoder.new_encoder_no_optimize import Encoder
from encoder.problem import Problem

p = Problem('data_set/j30.sm/j301_1.sm')
# p =Problem('test_2022.sm')

e = Encoder(p, 125)
for j in range(e.problem.njobs):
    print(
        f'Job {j}: ES{j} = {e.ES[j]}, LS{j} = {e.LS[j]}, EC{j} = {e.EC[j]}, LC{j} = {e.LC[j]},'
        f' length = {e.LS[j] - e.ES[j]}')
# e.encode()
# solver = Glucose3()
# for c in e.sat_model.clauses:
#     solver.add_clause(c)
#
# t = solver.solve()
# print(t)
# model = solver.get_model()
# # print(e.get_result(model))
# a  = e.get_result(model)
# for i in range(len(a)):
#     print(f'Job {i}: {a[i]}')
