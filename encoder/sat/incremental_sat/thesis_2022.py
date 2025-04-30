from encoder.sat.incremental_sat.incremental_sat_solver import IncrementalSATSolver


class ThesisMethod(IncrementalSATSolver):
    def _precedence_constraint(self):
        def start_clause():
            for i in range(1, self._problem.number_of_activities):
                clause = []
                for s in range(self._ES[i], self._LS[i] + 1):  # s in STW(i)
                    clause.append(self._start[(i, s)])
                self._sat_model.add_clause(clause)

        start_clause()
        for i in range(1, self._problem.number_of_activities):
            for j in self._problem.predecessors[i]:
                for s in range(self._ES[i], self._LS[i] + 1):  # s in STW(i)
                    clause = [-self._start[(i, s)]]
                    t = self._ES[j]
                    while t <= s - self._problem.durations[j] and t <= self._LS[j]:
                        clause.append(self._start[(j, t)])
                        t += 1
                    self._sat_model.add_clause(clause)
