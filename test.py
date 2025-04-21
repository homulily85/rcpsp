from encoder.problem import Problem
from encoder.sat.incremental_sat.staircase import StaircaseSATEncoder
from encoder.sat.incremental_sat.staircase_new import NewStaircaseSATEncoder

# p = Problem('data_set/data_set_test/test_2020.sm')
# p = Problem('data_set/pack/Pack004.rcp')
# p = Problem('data_set/data_set_test/test_2022.sm')
p = Problem('data_set/j30.sm/j301_1.sm')
e = StaircaseSATEncoder(p, 81, 0, enable_verify=True)
e.encode()
sat = e.solve()
while sat:
    print('Feasible with makespan:', e.makespan)
    print(e.get_solution())
    e.verify()
    print('Verify passed')
    e.decrease_makespan()
    e.solve()

else:
    print('Unfeasible with makespan:', e.makespan)
    print('Total time:', e.time_used)
