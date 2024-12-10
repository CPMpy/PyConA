import time
import cpmpy as cp
from cpmpy.transformations.get_variables import get_variables

from .qgen_core import *
from .qgen_obj import *
from ..utils import get_con_subset, restore_scope_values, Objectives


class PQGen(QGenBase):
    """
    PQGen function for query generation.
    This class implements the query generator from:
    Dimos Tsouros, Senne Berden, and Tias Guns. "Guided Bottom-Up Interactive Constraint Acquisition." CP, 2023
    """

    def __init__(self, ca_env: ActiveCAEnv = None, *, objective_function=None, time_limit=1, blimit=5000):
        """
        Initialize the PQGen with the given parameters.

        :param ca_env: The CA system.
        :param objective_function: The objective function for PQGen.
        :param time_limit: The time limit for query generation.
        :param blimit: The bias limit to start optimization.
        """
        super().__init__(ca_env, time_limit)
        self.partial = False
        if objective_function is None:
            objective_function = obj_max_viol
        self.obj = objective_function
        self.blimit = blimit

    @property
    def obj(self):
        """
        Get the objective of PQGen.

        :return: The objective function.
        """
        return self._obj

    @obj.setter
    def obj(self, obj):
        """
        Set the objective of PQGen.

        :param obj: The objective function to set.
        """
        assert obj in Objectives.qgen_objectives()
        self._obj = obj

    @property
    def blimit(self):
        """
        Get the bias limit to start optimization in PQGen.

        :return: The bias limit.
        """
        return self._blimit

    @blimit.setter
    def blimit(self, blimit):
        """
        Set the bias limit to start optimization in PQGen.

        :param blimit: The bias limit.
        """
        self._blimit = blimit

    def generate(self):
        """
        Generate a query using PQGen.

        :return: A set of variables that form the query.
        """
        # Start time (for the cutoff t)
        t0 = time.time()

        # Project down to only vars in scope of B
        Y = frozenset(get_variables(self.env.instance.bias))
        lY = list(Y)

        if len(Y) == len(self.env.instance.X):
            B = self.env.instance.bias
            Cl = self.env.instance.cl
        else:
            B = get_con_subset(self.env.instance.bias, Y)
            Cl = get_con_subset(self.env.instance.cl, Y)

        # If no constraints left in B, just return
        if len(B) == 0:
            return set()

        # sample from B using the probabilities -------------------
        # If no constraints learned yet, start by just generating an example in all the variables in Y
        if len(Cl) == 0:
            Cl = [cp.sum(Y) >= 1]

        if not self.partial and len(B) > self.blimit:

            m = cp.Model(Cl)
            flag = m.solve()  # no time limit to ensure convergence

            if flag and not all([c.value() for c in B]):
                return lY
            else:
                self.partial = True

        m = cp.Model(Cl)
        s = cp.SolverLookup.get("ortools", m)

        # We want at least one constraint to be violated to assure that each answer of the user
        # will lead to new information
        s += ~cp.all(B)

        # Solve first without objective (to find at least one solution)
        flag = s.solve()

        t1 = time.time() - t0
        if not flag or (t1 > self.time_limit):
            # UNSAT or already above time_limit, stop here --- cannot optimize
            return lY if flag else set()

        # Next solve will change the values of the variables in lY
        # so we need to return them to the original ones to continue if we don't find a solution next
        values = [x.value() for x in lY]

        # So a solution was found, try to find a better one now
        s.solution_hint(lY, values)
        try:
            objective = self.obj(B=B, ca_env=self.env)
        except:
            raise NotImplementedError(f"Objective given not implemented in PQGen: {self.obj} - Please report an issue")

        # Run with the objective
        s.maximize(objective)

        flag2 = s.solve(time_limit=(self.time_limit - t1))

        if flag2:
            return lY
        else:
            restore_scope_values(lY, values)
            return lY
