from encoder.lia.LIA_encoder import LIAEncoder
from encoder.problem import Problem
from encoder.sat.max_sat.MaxSAT_encoder import MaxSATEncoder

# p = Problem('data_set/pack/Pack004.rcp')
# p = Problem('data_set/data_set_test/test_2022.sm')
p = Problem('data_set/j60.sm/j609_2.sm')
e = MaxSATEncoder(p, 90, 80, enable_verify=True)
e.encode()
sat = e.solve()
if sat:
    print('Feasible with makespan:', e.makespan)
    print(e.get_solution())
    e.verify()
    print('Verify passed')
    # print('Number of variables:', e.sat_model.number_of_variable)
    # print('Number of clauses:', e.sat_model.number_of_clause)
else:
    print('Unfeasible with makespan:', e.makespan)
    print('Total time:', e.time_used)
