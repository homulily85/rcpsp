from encoder.sat.incremental_sat.Improved_resource_constraint import ImprovedResourceConstraint
from encoder.sat.incremental_sat.staircase import StaircaseMethod
from encoder.sat.incremental_sat.staircase_new import ImprovedStaircaseMethod
from encoder.sat.incremental_sat.thesis_2022 import ThesisMethod
from encoder.sat.max_sat.maxsat_solver import MaxSATSolver

e = ImprovedResourceConstraint('data_set/data_set_test/test_2020.sm', 3, 8, enable_verify=True)
e.encode()
model = e.solve()
while model:
    print('Feasible with makespan:', e.makespan)
    print(e.solution)
    e.verify()
    print('Verify passed')
    e.decrease_makespan()
    model = e.solve()

else:
    print('Unfeasible with makespan:', e.makespan)
    print('Total time:', e.time_used)

# lia = LIASolver('data_set/j30.sm/j301_1.sm')
# print(lia.solve())
# print(lia.solution)
