from abc import ABC, abstractmethod
import cpmpy as cp

from ..ca_environment.active_ca import ActiveCAEnv
from .findc_obj import findc_obj_splithalf
from ..utils import get_scope, restore_scope_values, Objectives


class FindCBase(ABC):
    """
    Abstract class interface for FindC implementations.
    """

    def __init__(self, ca_system: ActiveCAEnv = None, time_limit=0.2, *, findc_obj=findc_obj_splithalf):
        """
        Initialize the FindCBase class.

        :param ca_system: The constraint acquisition system.
        :param time_limit: The time limit for findc query generation.
        :param findc_obj: The function to use for findc object, default is findc_obj_splithalf.
        """
        self.ca = ca_system
        self._time_limit = time_limit
        self.obj = findc_obj

    @abstractmethod
    def run(self, scope):
        """
        Method that all FindC implementations must implement.

        :param scope: The scope in which we search for a constraint.
        """
        assert self.ca is not None
        raise NotImplementedError

    @property
    def ca(self):
        """
        Get the constraint acquisition system.

        :return: The constraint acquisition system.
        """
        return self._ca

    @ca.setter
    def ca(self, ca_system: ActiveCAEnv = None):
        """
        Set the constraint acquisition system.

        :param ca_system: The constraint acquisition system.
        """
        if ca_system is not None:
            self._ca = ca_system
            if self._ca.findc != self:
                self._ca.findc = self

    @property
    def time_limit(self):
        """
        Get the time limit for findc query generation.

        :return: The time limit.
        """
        return self._time_limit

    @time_limit.setter
    def time_limit(self, time_limit):
        """
        Set the time limit for findc query generation.

        :param time_limit: The time limit.
        """
        self._time_limit = time_limit

    @property
    def obj(self):
        """
        Get the objective of pqgen.

        :return: The objective.
        """
        return self._obj

    @obj.setter
    def obj(self, obj):
        """
        Set the objective of pqgen.

        :param obj: The objective to set.
        """
        assert obj in Objectives.findc_objectives()
        self._obj = obj

    def generate_findc_query(self, L, delta):
        """
        Generate a findc query.

        Constraints from B are taken into account as soft constraints.
        The objective function used is the one presented in the same paper, doing a dichotomy search on delta.
        Changes directly the values of the variables.

        :param L: Learned network in the given scope.
        :param delta: Candidate constraints in the given scope.
        :return: Boolean value representing a success or failure on the generation.
        """
        tmp = cp.Model(L)

        sat = sum([c for c in delta])  # Get the amount of satisfied constraints from B

        # At least 1 violated and at least 1 satisfied:
        # We want this to assure that each answer of the user will reduce the set of candidates
        # If all are violated, we already know that the example will be a non-solution due to previous answers!
        tmp += sat < len(delta)
        tmp += sat > 0

        # Try first without objective
        s = cp.SolverLookup.get("ortools", tmp)
        flag = s.solve()

        if not flag:
            # UNSAT, stop here
            return False

        Y = get_scope(delta[0])
        Y = list(dict.fromkeys(Y))  # Remove duplicates

        # Next solve will change the values of the variables in the lY2
        # So we need to return them to the original ones to continue if we don't find a solution next
        values = [x.value() for x in Y]

        # So a solution was found, try to find a better one now
        s.solution_hint(Y, values)

        objective = self.obj(sat, delta, ca_system=self.ca)
        # Run with the objective
        s.maximize(objective)  # We want to try and do it like a dichotomic search

        flag2 = s.solve(time_limit=self.time_limit)

        if not flag2:
            restore_scope_values(Y, values)
            return flag

        else:
            return flag2
