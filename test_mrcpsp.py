from src.mrcpsp.problem import MRCPSPProblem
from src.mrcpsp.solver import MRCPSPSolver

if __name__ == "__main__":
    # p = MRCPSPProblem.from_file('data_set/mrcpsp/MMLIB50/J501_1.mm')
    p = MRCPSPProblem.from_file('data_set/mrcpsp/j30.mm/j3010_2.mm')
    s = MRCPSPSolver(p)
    s.encode()
    print(s.solve(find_optimal=True, time_limit=600))
    print(s.get_makespan())
    print(s.get_schedule())
    s.get_graph(save_to_a_file=True)
    print(s.get_statistics())
    s.verify()