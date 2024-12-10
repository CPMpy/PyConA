from abc import abstractmethod
from ..ca_environment.active_ca import ActiveCAEnv
import cpmpy as cp
from cpmpy.solvers.solver_interface import ExitStatus

from .qgen_core import QGenBase


class QGen(QGenBase):
    """
    Baseline for QGen implementations
    """

    def __init__(self, ca_env: ActiveCAEnv = None, time_limit=600):
        """
        Initialize the QGen class.

        :param ca_env: An instance of ActiveCAEnv, default is None.
        :param time_limit: Time limit for the solver, default is 600 seconds.
        """
        super().__init__(ca_env, time_limit)


    @abstractmethod
    def generate(self):
        """
        A basic version of query generation for small problems. May lead
        to premature convergence, so generally not used.

        :return: A set of variables that form the query.
        """
        if len(self.env.instance.bias) == 0:
            return False

        # B are taken into account as soft constraints that we do not want to satisfy (i.e., that we want to violate)
        m = cp.Model(self.env.instance.cl)  # could use to-be-implemented m.copy() here...

        # Get the amount of satisfied constraints from B
        objective = sum([c for c in self.env.instance.bias])

        # We want at least one constraint to be violated to assure that each answer of the
        # user will reduce the set of candidates
        m += objective < len(self.env.instance.bias)

        s = cp.SolverLookup.get("ortools", m)
        flag = s.solve(time_limit=self.time_limit)

        if not flag:
            if s.cpm_status.exitstatus == ExitStatus.UNKNOWN:
                self.env.converged = 0
                return set()

        return self.env.instance.X
