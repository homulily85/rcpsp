from encoder.staircase import StaircaseEncoder
from encoder.problem import Problem

# p = Problem('data_set/pack_d/Pack_d001.rcp')
p = Problem('data_set/j30.sm/j301_1.sm')
# p = Problem('data_set_test/test_2020.sm')

e = StaircaseEncoder(p, 43)
e.encode()
sat = e.solve()
while sat:
    print('Feasible with makespan:', e.makespan)
    print(e.get_result())
    e.verify()
    print('Verify passed')
    e.decrease_makespan()
    sat = e.solve()

else:
    print('Unfeasible with makespan:', e.makespan)
    print('Total time:', e.time_used)
