from encoder.problem import Problem
from encoder.staircase import StaircaseSATEncoder

p = Problem('data_set/j30.sm/j301_2.sm')
# p = Problem('data_set/data_set_test/test_2020.sm')
# p = Problem('data_set/data_set_test/test_2022.sm')
e = StaircaseSATEncoder(p, 47)
e.encode()
sat = e.solve()
while sat:
    print('Feasible with makespan:', e.makespan)
    print(e.get_result ())
    e.verify()
    print('Verify passed')
    print('Number of variables:', e.sat_model.number_of_variable)
    print('Number of clauses:', e.sat_model.number_of_clause)
    e.decrease_makespan()
    sat = e.solve()
else:
    print('Unfeasible with makespan:', e.makespan)
    print('Total time:', e.time_used)
