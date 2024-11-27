import cpmpy as cp
from cpmpy.transformations.normalize import toplevel_list

from .oracle import Oracle
from ..utils import get_con_subset, check_value


class ConstraintOracle(Oracle):
    """
    Oracle based on the target set of constraints, which is used to answer the given queries
    """

    def __init__(self, constraints):
        """
        Initialize the ConstraintOracle with the target set of constraints.

        :param constraints: The set of constraints C_T used for answering queries.
        """
        super().__init__()
        self.constraints = constraints

    @property
    def constraints(self):
        """
        get the constraints of the ConstraintOracle
        :return: self._constraints
        """
        return self._constraints

    @constraints.setter
    def constraints(self, constraints):
        """
        setter for the constraints property
        """
        constraints = toplevel_list(constraints)
        self._constraints = constraints

    def answer_membership_query(self, Y):
        """
        Answer a membership query.

        Determines whether the given assignment on Y is a solution or not using the constraints of the problem.

        :param Y: The input values to be checked for membership.
        :return: A boolean indicating a positive or negative answer.
        """

        # Need the oracle to answer based only on the constraints with a scope that is a subset of Y
        suboracle = get_con_subset(self.constraints, Y)
        # Check if at least one constraint is violated or not
        return all([check_value(c) for c in suboracle])

    def answer_recommendation_query(self, c):
        """
        Answer a recommendation query by checking if the recommended constraint is part of the target set of
        constraints, or logically implied by the constraints in the target set of constraints.

        :param c: The recommended constraint.
        :return: A boolean indicating if the recommended constraint is in the set of constraints.
        """
        # Check if the recommended constraint is in the set of constraints or implied by them
        m = cp.Model(self.constraints)
        m += ~c
        return not m.solve()

    def answer_generalization_query(self, C):
        """
        Answer a generalization query by checking if the set of constraints C is part of C_T.

        :param C: The constraints representing the generalization.
        :return: A boolean indicating if the generalization is correct.
        """
        # For simplicity, we assume that if all constraints in C are in by C_T, the generalization is correct.
        return all(constraint in set(self.constraints) for constraint in C)
