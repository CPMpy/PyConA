import math
import time
import cpmpy as cp
from cpmpy.solvers.solver_interface import ExitStatus

from .qgen_core import *
from ..utils import get_con_subset, get_min_arity, get_scope


class TQGen(QGenBase):
    """
    TQ-Gen function for query generation.

    This class implements the query generator from:
    Ait Addi, Hajar, et al. "Time-bounded query generator for constraint acquisition." CPAIOR, 2018
    """

    def __init__(self, ca_env: ActiveCAEnv = None, *, time_limit=2, tau=0.2, alpha=0.8, l=None):
        """
        Initialize the TQGen with the given parameters.

        :param ca_env: The CA environment used.
        :param time_limit: Overall time limit.
        :param tau: Solving timeout.
        :param alpha: Reduction factor.
        :param l: Expected query size.
        """
        super().__init__(ca_env, time_limit)
        self._tau = tau
        self._alpha = alpha
        self._lamda = l

    @property
    def tau(self):
        """Get the tau parameter of TQGen."""
        return self._tau

    @tau.setter
    def tau(self, tau):
        """Set the tau parameter of TQGen."""
        self._tau = tau

    @property
    def alpha(self):
        """Get the alpha parameter of TQGen."""
        return self._alpha

    @alpha.setter
    def alpha(self, alpha):
        """Set the alpha parameter of TQGen."""
        self._alpha = alpha

    @property
    def lamda(self):
        """Get the lamda parameter of TQGen."""
        return self._lamda

    @lamda.setter
    def lamda(self, lamda):
        """Set the lamda parameter of TQGen."""
        self._lamda = lamda

    def generate(self, Y=None):
        """
        Generate a query using TQGen.

        :return: A list of variables that form the query.
        """
        if Y is None:
            Y = self.env.instance.X
        assert isinstance(Y, list), "When generating a query, Y must be a list of variables"

        if self._lamda is None:
            self._lamda = len(self._env.instance.X)

        ttime = 0
        bias = get_con_subset(self.env.instance.bias, Y)
        cl = get_con_subset(self.env.instance.cl, Y)

        while (ttime < self.time_limit) and (len(bias) > 0):
            t = min([self.tau, self.time_limit - ttime])
            l = max([self.lamda, get_min_arity(bias)])

            Y2 = find_suitable_vars_subset2(l, bias, Y)

            B = get_con_subset(bias, Y2)
            Cl = get_con_subset(cl, Y2)

            m = cp.Model(Cl)
            s = cp.SolverLookup.get("ortools", m)

            # Create indicator variables upfront
            V = cp.boolvar(shape=(len(B),))
            s += (V != B)

            # We want at least one constraint to be violated
            s += sum(V) > 0

            t_start = time.time()
            flag = s.solve(time_limit=t)
            ttime = ttime + (time.time() - t_start)

            if flag:
                return Y

            if s.ort_status == ExitStatus.UNSATISFIABLE:
                self.env.add_to_cl(B)
            else:
                l = int((self.alpha * l) // 1)  # //1 to round down

        if len(bias) > 0:
            self.env.converged = 0

        return []

    def adjust(self, l, a, answer):
        """
        Adjust the number of variables taken into account in the next iteration of TQGen.

        :param l: Current number of variables.
        :param a: Adjustment factor.
        :param answer: Boolean indicating whether the previous query was successful.
        :return: Adjusted number of variables.
        """
        if answer:
            l = min([int(math.ceil(l / a)), len(self.env.instance.X)])
        else:
            l = int((a * l) // 1)  # //1 to round down
        if l < get_min_arity(self.env.instance.bias):
            l = 2
        return l


def find_suitable_vars_subset(l, B, Y):
    """
    Find a suitable subset of variables.

    :param l: Number of variables to select.
    :param B: Bias constraints.
    :param Y: List of variables.
    :return: A subset of variables.
    """
    if len(Y) < get_min_arity(B):
        return Y

    hashY = [hash(y) for y in Y]

    x = cp.boolvar(shape=(len(Y),))
    v = cp.boolvar(shape=(len(B),))

    model = cp.Model()

    model += [cp.all(x[hashY.index(hash(scope_var))] for scope_var in get_scope(B[i])).implies(v[i]) for i in
              range(len(B))]
    model += sum(v) > 0
    Y1_size = sum(x)

    model += Y1_size == l
    model.solve()

    Y1 = [Y[i] for i in range(len(Y)) if x[i].value()]

    return Y1


def find_suitable_vars_subset2(l, B, Y):
    """
    Find a suitable subset of variables (alternative method).

    :param l: Number of variables to select.
    :param B: Bias constraints.
    :param Y: List of variables.
    :return: A subset of variables.
    """
    if len(Y) <= get_min_arity(B) or len(B) < 1:
        return Y

    scope = get_scope(B[0])
    Y_prime = list(set(Y) - set(scope))

    l2 = int(l) - len(scope)

    if l2 > 0:
        Y1 = Y_prime[:l2]
    else:
        Y1 = []

    [Y1.append(y) for y in scope]

    return Y1