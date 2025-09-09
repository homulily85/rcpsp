from src.mrcpsp_solver import MRCPSPProblem, MRCPSPSolver

if __name__ == "__main__":
    # p = MRCPSPProblem('data_set/mrcpsp/test.mm')
    p = MRCPSPProblem('data_set/mrcpsp/j30.mm/j3010_8.mm')
    # p = RCPSPProblem('data_set/rcpsp/pack_d/Pack_d001.rcp')
    # p = RCPSPProblem('data_set/rcpsp/test/test_2022.sm')
    # p = RCPSPProblem('data_set/rcpsp/j30.sm/j302_10.sm')
    # p = RCPSPProblem('data_set/rcpsp/j90.sm/j901_1.sm')
    s = MRCPSPSolver(p, 2, 32)
    s.encode()
    print(s.solve(find_optimal=True, time_limit=600))
    print(s.get_makespan())
    print(s.get_schedule())
    s.get_graph(save_to_a_file=True)
    print(s.get_statistics())
    s.verify()
