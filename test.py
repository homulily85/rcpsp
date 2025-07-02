from src.solver import Problem, RCPSPSolver

# p = Problem('data_set/pack_d/Pack_d046.rcp', 'pack')
p = Problem('data_set/pack/Pack001.rcp', 'pack')
s = RCPSPSolver(p, 'sat')
s.encode()
s.solve(find_optimal=True)
s.verify()
s.get_schedule(get_graph=True, save_graph_to_file=True)
print(s.get_statistics())
