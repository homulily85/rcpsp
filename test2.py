from encoder.SAT_model import SATModel


class Test:
    def __init__(self):
        self.sat_model = SATModel()

    def SCPB_at_most_K(self, literals: list[int], weights: list[int], k: int):
        if len(literals) != len(weights):
            raise ValueError("The length of literals and weights must be the same.")
        self.sat_model.set_start_value(literals[-1])
        always_false = []
        for w in weights:
            if w > k:
                self.sat_model.add_clause([-literals[weights.index(w)]])
                print([-literals[weights.index(w)]])
                always_false.append(weights.index(w))

        for i in range(len(always_false)):
            literals.remove(literals[always_false[i]])
            weights.remove(weights[always_false[i]])

        bit = {}
        for i in range(len(literals)):
            if i == 0:
                bit[literals[i]] = [self.sat_model.get_new_var() for _ in
                                    range(min(k, weights[i]))]
            elif i + 1 < k:
                bit[literals[i]] = [self.sat_model.get_new_var() for _ in
                                    range(
                                        min(k, sum([weights[j] for j in range(0, i + 1)])))]
            else:
                bit[literals[i]] = [self.sat_model.get_new_var() for _ in range(k)]

        print(bit)

        for i in range(len(literals) - 1):
            for j in range(weights[i]):
                self.sat_model.add_clause([-literals[i], bit[literals[i]][j]])
                print([-literals[i], bit[literals[i]][j]])

        for i in range(1, len(literals) - 1):
            for j in range(0, len(bit[literals[i - 1]])):
                self.sat_model.add_clause([-bit[literals[i - 1]][j], bit[literals[i]][j]])
                print([-bit[literals[i - 1]][j], bit[literals[i]][j]])

        for i in range(1, len(literals) - 1):
            temp = []
            try:
                for j in range(0, len(bit[literals[i - 1]])):
                    temp.append([-literals[i], -bit[literals[i - 1]][j],
                                 bit[literals[i]][j + weights[i]]])
                for c in temp:
                    self.sat_model.add_clause(c)
                    print(c)
            except IndexError:
                continue

        for i in range(1, len(literals)):
            try:
                self.sat_model.add_clause(
                    [-literals[i], -bit[literals[i - 1]][k - weights[i]]])
                print([-literals[i], -bit[literals[i - 1]][k - weights[i]]])
            except IndexError:
                continue


def main():
    t = Test()
    t.SCPB_at_most_K([1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 4, 10, 0, 3, 4, 0, 6, 0, 4], 12)
    t.sat_model.add_clause([1])
    t.sat_model.add_clause([2])
    t.sat_model.add_clause([3])
    t.sat_model.add_clause([4])
    t.sat_model.add_clause([5])
    t.sat_model.add_clause([6])
    t.sat_model.add_clause([7])
    t.sat_model.add_clause([8])
    t.sat_model.solver.solve()
    print(t.sat_model.solver.get_model())


if __name__ == "__main__":
    main()
