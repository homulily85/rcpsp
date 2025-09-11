from src.rcpsp.problem import RCPSPProblem
from src.rcpsp.solver import RCPSPSolver

if __name__ == "__main__":
    p = RCPSPProblem.from_file('data_set/rcpsp/pack/Pack001.rcp')
    # p = RCPSPProblem('data_set/rcpsp/pack_d/Pack_d001.rcp')
    # p = RCPSPProblem('data_set/rcpsp/test/test_2022.sm')
    # p = RCPSPProblem('data_set/rcpsp/j30.sm/j302_10.sm')
    # p = RCPSPProblem('data_set/rcpsp/j90.sm/j901_1.sm')
    s = RCPSPSolver(p)
    s.encode()
    print(s.solve(find_optimal=True, time_limit=600))
    print(s.get_schedule()[-1])
    s.get_graph(save_to_a_file=True)
    s.verify()
    print(s.get_statistics())