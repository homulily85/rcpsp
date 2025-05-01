from encoder.model.weighted_at_most_k_model import WeightedAtMostKModel
from encoder.sat.incremental_sat.staircase_new import ImprovedStaircaseMethod


class ImprovedResourceConstraintImprovedStaircaseBased(ImprovedStaircaseMethod):
    def _create_variable(self):
        for i in range(self._problem.number_of_activities):
            for t in range(self._ES[i],
                           self._LS[i] + 1):  # t in STW(i) (start time window of activity i)
                self._start[(i, t)] = self.sat_model.create_new_variable()

    def _resource_constraint(self):
        """Add resource constraints to the SAT model."""
        for t in range(self._upper_bound + 1):
            for r in range(self._problem.number_of_resources):
                pb_constraint = WeightedAtMostKModel(self.sat_model, self._problem.capacities[r])
                for i in range(1, self._problem.number_of_activities):
                    temp_max = -1
                    temp_min = float("inf")
                    for s in range(self._ES[i], self._LS[i] + 1):
                        if s <= t < s + self._problem.durations[i]:
                            if s > temp_max:
                                temp_max = s
                            if s < temp_min:
                                temp_min = s

                    if temp_max != -1 and temp_min != float("inf"):
                        pb_constraint.add_term(
                            self._get_forward_staircase_register(i, temp_min, temp_max + 1),
                            self._problem.requests[i][r])
                pb_constraint.encode()

    def encode(self):
        """Encode the problem into SAT clauses."""
        self._start_time_for_job_0()
        self._precedence_constraint()
        self._resource_constraint()
