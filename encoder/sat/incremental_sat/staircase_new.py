from encoder.sat.incremental_sat.staircase import StaircaseMethod


class ImprovedStaircaseMethod(StaircaseMethod):
    """
    Class for encoding the Resource-Constrained Project Scheduling Problem (RCPSP) using the improved staircase method.
    """

    # def _precedence_constraint(self):
    #     for predecessor in range(1, self._problem.number_of_activities):
    #         for successor in self._problem.successors[predecessor]:
    #             # Successor can only start at one time
    #             self._sat_model.add_clause(
    #                 [self._get_forward_staircase_register(successor, self._ES[successor],
    #                                                       self._LS[successor] + 1)])
    #             # Predecessor can only start at one time
    #             self._sat_model.add_clause(
    #                 [self._get_forward_staircase_register(predecessor, self._ES[predecessor],
    #                                                       self._LS[predecessor] + 1)])
    #
    #             # Precedence constraint
    #             for k in range(self._ES[successor], self._LS[successor] + 1):
    #                 first_half = self._get_forward_staircase_register(successor,
    #                                                                   self._ES[successor],
    #                                                                   k + 1)
    #
    #                 if first_half is None or k - self._problem.durations[predecessor] + 1 >= self._LS[
    #                     predecessor] + 1:
    #                     continue
    #
    #                 self._sat_model.add_clause(
    #                     [-first_half,
    #                      -self._start[predecessor, k - self._problem.durations[predecessor] + 1]])
    #             else:
    #                 for k in range(self._LS[successor] - self._problem.durations[predecessor] + 2,
    #                                self._LS[predecessor] + 1):
    #                     self._sat_model.add_clause(
    #                         [-self._get_forward_staircase_register(successor, self._ES[successor],
    #                                                                self._LS[successor] + 1),
    #                          -self._start[predecessor, k]])

    def _precedence_constraint(self):
        for pred in range(1, self._problem.number_of_activities):
            for succ in self._problem.successors[pred]:
                # Ensure each job has exactly one start: use backward staircase over its entire window
                self._sat_model.add_clause([
                    self._get_backward_staircase_register(succ,
                                                          self._ES[succ],
                                                          self._LS[succ] + 1)
                ])
                self._sat_model.add_clause([
                    self._get_backward_staircase_register(pred,
                                                          self._ES[pred],
                                                          self._LS[pred] + 1)
                ])

                for k in range(self._ES[succ], self._LS[succ] + 1):
                    second_half = self._get_backward_staircase_register(pred,
                                                                        k - self._problem.durations[
                                                                            pred] + 1,
                                                                        self._LS[pred] + 1)

                    if second_half is None or k - self._problem.durations[pred] + 1 >= self._LS[
                        pred] + 1:
                        continue

                    self._sat_model.add_clause([-second_half, -self._start[succ, k]])
                else:
                    for k in range(self._LS[succ] - self._problem.durations[pred] + 2,
                                   self._LS[pred] + 1):
                        self._sat_model.add_clause(
                            [-self._get_backward_staircase_register(succ, self._ES[succ],
                                                                    self._LS[succ] + 1),
                             -self._start[pred, k]])