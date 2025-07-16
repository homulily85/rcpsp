from src.solver import Problem, RCPSPSolver

# p = Problem('data_set/pack_d/Pack_d046.rcp', 'pack')
p = Problem('data_set/pack/Pack001.rcp', 'pack')
# p = Problem('data_set/data_set_test/test_2022.sm', 'psplib')
# p = Problem('data_set/j30.sm/j301_1.sm', 'psplib')
s = RCPSPSolver(p, 'sat')
s.encode()
print(s.solve(find_optimal=True,time_limit=600))
# print(s.get_schedule(get_graph=True, save_graph_to_file=True)[-1])
s.verify()
print(s.get_statistics())