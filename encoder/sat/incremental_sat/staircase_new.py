from encoder.sat.incremental_sat.staircase import StaircaseSATEncoder


class NewStaircaseSATEncoder(StaircaseSATEncoder):
    def _precedence_constraint(self):
        for predecessor in range(1, self.problem.njobs):
            for successor in self.problem.successors[predecessor]:
                # Get forward staircase registers once for reuse
                successor_register = self._get_forward_staircase_register(successor, self.ES[successor],
                                                                          self.LS[successor] + 1)
                predecessor_register = self._get_forward_staircase_register(predecessor, self.ES[predecessor],
                                                                            self.LS[predecessor] + 1)
                
                # Skip if registers aren't valid
                if successor_register is None or predecessor_register is None:
                    continue
                    
                # Successor can only start at one time
                self.sat_model.add_clause([successor_register])
                # Predecessor can only start at one time
                self.sat_model.add_clause([predecessor_register])

                # Precedence constraint
                for k in range(self.ES[successor], self.LS[successor] + 1):
                    first_half = self._get_forward_staircase_register(successor,
                                                                      self.ES[successor],
                                                                      k + 1)

                    if first_half is None:
                        continue
                        
                    pred_time = k - self.problem.durations[predecessor] + 1
                    # Skip if predecessor time is out of range
                    if pred_time >= self.LS[predecessor] + 1:
                        continue
                        
                    # Check if predecessor can start at this time
                    if pred_time >= self.ES[predecessor]:
                        self.sat_model.add_clause(
                            [-first_half, -self.start[predecessor, pred_time]])
                
                # Handle remaining predecessor times that would violate the precedence constraint
                # This is separate from the loop above and not an "else" clause
                latest_valid_pred_start = self.LS[successor] - self.problem.durations[predecessor] + 1
                for k in range(latest_valid_pred_start + 1, self.LS[predecessor] + 1):
                    if k < self.ES[predecessor]:
                        continue
                    self.sat_model.add_clause(
                        [-successor_register, -self.start[predecessor, k]])
