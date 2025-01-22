import unittest

from pysat.solvers import Glucose3

from Encoder import Encoder
from Problem import Problem


class MyTestCase(unittest.TestCase):
    def test_calc_time_windows(self):
        p = Problem('j301_1.sm')

        e = Encoder(p, (8, 8))

        e.encode()
        self.assertEqual([0, 0, 0, 2, 1, 2, 3, 5, 6], e.ES)
        self.assertEqual([0, 2, 1, 3, 2, 5, 5, 6, 6], e.EC)
        self.assertEqual([2, 4, 4, 5, 5, 7, 7, 8, 8], e.LC)
        self.assertEqual([2, 2, 3, 4, 4, 4, 5, 7, 8], e.LS)

    def test_result(self):
        p = Problem('j301_1.sm')

        e = Encoder(p, (100,222))
        e.encode()
        solver = Glucose3()

        for clause in e.sat_model.clauses:
            solver.add_clause(clause)

        self.assertTrue(solver.solve())

        # self.assertEqual([0, 0, 2, 3, 4, 4, 5, 7, 8], e.get_result(solver.get_model()))


if __name__ == '__main__':
    unittest.main()
