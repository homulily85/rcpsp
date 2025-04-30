from encoder.sat.incremental_sat.staircase import StaircaseMethod


class ImprovedStaircaseMethod(StaircaseMethod):
    """
    Class for encoding the Resource-Constrained Project Scheduling Problem (RCPSP) using the improved staircase method.
    """
    def _precedence_constraint(self):
        for predecessor in range(1, self._problem.number_of_activities):
            for successor in self._problem.successors[predecessor]:
                # Successor can only start at one time
                self._sat_model.add_clause(
                    [self._get_forward_staircase_register(successor, self._ES[successor],
                                                          self._LS[successor] + 1)])
                # Predecessor can only start at one time
                self._sat_model.add_clause(
                    [self._get_forward_staircase_register(predecessor, self._ES[predecessor],
                                                          self._LS[predecessor] + 1)])

                # Precedence constraint
                for k in range(self._ES[successor], self._LS[successor] + 1):
                    first_half = self._get_forward_staircase_register(successor,
                                                                      self._ES[successor],
                                                                      k + 1)

                    if first_half is None or k - self._problem.durations[predecessor] + 1 >= self._LS[
                        predecessor] + 1:
                        continue

                    self._sat_model.add_clause(
                        [-first_half,
                         -self._start[predecessor, k - self._problem.durations[predecessor] + 1]])
                else:
                    for k in range(self._LS[successor] - self._problem.durations[predecessor] + 2,
                                   self._LS[predecessor] + 1):
                        self._sat_model.add_clause(
                            [-self._get_forward_staircase_register(successor, self._ES[successor],
                                                                   self._LS[successor] + 1),
                             -self._start[predecessor, k]])

