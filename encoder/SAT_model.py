class SATModel:
    def __init__(self):
        self.number_of_variable = 0
        self.clauses: list[list[int]] = []

    def get_new_var(self) -> int:
        self.number_of_variable += 1
        return self.number_of_variable
