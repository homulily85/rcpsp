from pysat.pb import *

from encoder.model.sat_model import Model, SATModel, MaxSATModel


class WeightedAtMostKModel:
    """
    Class to encode a weighted at most k constraint.
    """

    def __init__(self, model: Model, bound: int):
        """
        Initialize the WeightedAtMostKModel.
        :param model: Model to be used for encoding.
        :type model: Model
        :param bound: The bound for the weighted at most k constraint.
        :type bound: int
        """
        self.__literals = []
        self.__weights = []
        self.__bound = bound
        self.__model = model

    def add_term(self, literal: int, weight: int):
        """
        Add a term to the weighted at most k constraint.
        :param literal: ID of the literal.
        :type literal: int
        :param weight: Weight of the literal.
        :type weight: int
        """
        self.__literals.append(literal)
        self.__weights.append(weight)

    def encode(self):
        """
        Encode the weighted at most k constraint into the model.
        """
        cnf = PBEnc.leq(lits=self.__literals, weights=self.__weights, bound=self.__bound,
                        top_id=self.__model.number_of_variables, encoding=EncType.bdd).clauses

        # cnf can be empty
        if not cnf:
            return
        new_variable_max_index = -1
        for clause in cnf:
            for var in clause:
                if abs(var) > new_variable_max_index:
                    new_variable_max_index = abs(var)
        if new_variable_max_index == -1:
            return

        self.__model.number_of_variables = max(new_variable_max_index,
                                               self.__model.number_of_variables)
        self.__model.number_of_pb_clauses += len(cnf)

        if isinstance(self.__model, SATModel):
            for clause in cnf:
                self.__model.add_clause(clause)

        elif isinstance(self.__model, MaxSATModel):
            for clause in cnf:
                self.__model.add_hard_clause(clause)
