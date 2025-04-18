from encoder.lia.LIA_encoder import LIAEncoder
from encoder.problem import Problem
from encoder.sat.incremental_sat.staircase_new import NewStaircaseSATEncoder
from encoder.sat.max_sat.MaxSAT_encoder import MaxSATEncoder

# p = Problem('data_set/pack/Pack004.rcp')
# p = Problem('data_set/data_set_test/test_2022.sm')
p = Problem('data_set/j30.sm/j301_1.sm')
e = MaxSATEncoder(p, 43, 43, enable_verify=True)
e.encode()
sat = e.solve()
while sat:
    print('Feasible with makespan:', e.makespan)
    print(e.get_solution())
    e.verify()
    print('Verify passed')
    e.decrease_makespan()
    sat = e.solve()
    # print('Number of variables:', e.sat_model.number_of_variable)
    # print('Number of clauses:', e.sat_model.number_of_clause)
else:
    print('Unfeasible with makespan:', e.makespan)
    print('Total time:', e.time_used)
