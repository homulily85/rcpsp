from encoder.staircase import StaircaseEncoder
from encoder.problem import Problem

p = Problem('data_set/j90.sm/j905_9.sm')
# p = Problem('data_set_test/test_2020.sm')

e = StaircaseEncoder(p, 115)
e.encode()
sat = e.solve(timeout=2)
print(sat)
while sat:
    print('Feasible with makespan:', e.makespan)
    e.decrease_makespan()
    sat = e.solve()

else:
    print('Unfeasible with makespan:', e.makespan)
    print('Total time:', e.time_used)
