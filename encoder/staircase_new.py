from encoder.staircase import StaircaseSATEncoder


class NewStaircaseSATEncoder(StaircaseSATEncoder):
    def _precedence_constraint(self):
        for predecessor in range(1, self.problem.njobs):
            for successor in self.problem.successors[predecessor]:
                # Successor can only start at one time
                self.sat_model.add_clause(
                    [self._get_forward_staircase_register(successor, self.ES[successor],
                                                          self.LS[successor] + 1)])
                # Predecessor can only start at one time
                self.sat_model.add_clause(
                    [self._get_forward_staircase_register(predecessor, self.ES[predecessor],
                                                          self.LS[predecessor] + 1)])

                # Precedence constraint
                for k in range(self.ES[successor], self.LS[successor] + 1):
                    first_half = self._get_forward_staircase_register(successor,
                                                                      self.ES[successor],
                                                                      k + 1)

                    if first_half is None or k - self.problem.durations[predecessor] + 1 >= self.LS[
                        predecessor] + 1:
                        continue

                    self.sat_model.add_clause(
                        [-first_half,
                         -self.start[predecessor, k - self.problem.durations[predecessor] + 1]])
                else:
                    for k in range(self.LS[successor] - self.problem.durations[predecessor] + 2,
                                   self.LS[predecessor] + 1):
                        self.sat_model.add_clause(
                            [-self._get_forward_staircase_register(successor, self.ES[successor],
                                                                   self.LS[successor] + 1),
                             -self.start[predecessor, k]])

