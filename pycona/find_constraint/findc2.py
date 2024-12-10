import cpmpy as cp

from ..ca_environment.active_ca import ActiveCAEnv
from .utils import get_max_conjunction_size, get_delta_p
from .findc_core import FindCBase
from .utils import join_con_net
from ..utils import restore_scope_values, get_con_subset, check_value


class FindC2(FindCBase):
    """
    This is the version of the FindC function that was presented in
    Bessiere, Christian, et al., "Learning constraints through partial queries", AIJ 2023

    This function works also for non-normalised target networks!
    """
    # TODO optimize to work better (probably only needs to make better the generate_find_query2)

    def __init__(self, ca_env: ActiveCAEnv = None, time_limit=0.2, findscope=None):
        """
        Initialize the FindC2 class.

        :param ca_env: The constraint acquisition environment.
        :param time_limit: The time limit for findc query generation.
        :param findscope: The function to find the scope.
        """
        super().__init__(ca_env, time_limit)
        self._findscope = findscope

    @property
    def findscope(self):
        """
        Get the findscope function to be used.

        :return: The findscope function.
        """
        return self._findscope

    @findscope.setter
    def findscope(self, findscope):
        """
        Set the findscope function to be used.

        :param findscope: The findscope function.
        """
        self._findscope = findscope

    def run(self, scope):
        """
        Run the FindC2 algorithm.

        :param scope: The scope in which we search for a constraint.
        :return: The constraint found.
        """
        assert self.ca is not None

        # Initialize delta
        delta = get_con_subset(self.ca.instance.bias, scope)
        delta = join_con_net(delta, [c for c in delta if check_value(c) is False])

        # We need to take into account only the constraints in the scope we search on
        sub_cl = get_con_subset(self.ca.instance.cl, scope)

        scope_values = [x.value() for x in scope]

        while True:

            # Try to generate a counter example to reduce the candidates
            if self.generate_findc_query(sub_cl, delta) is False:

                # If no example could be generated
                # Check if delta is the empty set, and if yes then collapse
                if len(delta) == 0:
                    raise Exception("Collapse, the constraint we seek is not in B")

                restore_scope_values(scope, scope_values)

                # Return random c in delta otherwise (if more than one, they are equivalent w.r.t. C_l)
                return delta[0]

            self.ca.metrics.increase_findc_queries()

            if self.ca.ask_membership_query(scope):
                # delta <- delta \setminus K_{delta}(e)
                delta = [c for c in delta if check_value(c) is not False]
                [self.ca.remove_from_bias(c) for c in delta if check_value(c) is False]

            else:  # user says UNSAT
                # delta <- joint(delta,K_{delta}(e))

                kappaD = [c for c in delta if check_value(c) is False]

                scope2 = self.ca.run_find_scope(list(scope), kappaD)  # TODO: replace with real findscope arguments when done!

                if len(scope2) < len(scope):
                    self.run(scope2)
                else:
                    delta = join_con_net(delta, kappaD)

    def generate_findc_query(self, L, delta):
        # TODO: optimize to work better
        """
        Changes directly the values of the variables

        :param L: learned network in the given scope
        :param delta: candidate constraints in the given scope
        :return: Boolean value representing a success or failure on the generation
        """

        tmp = cp.Model(L)

        max_conj_size = get_max_conjunction_size(delta)
        delta_p = get_delta_p(delta)

        p = cp.intvar(0, max_conj_size)
        kappa_delta_p = cp.intvar(0, len(delta), shape=(max_conj_size,))
        p_soft_con = cp.boolvar(shape=(max_conj_size,))

        for i in range(max_conj_size):
            tmp += kappa_delta_p[i] == sum([c for c in delta_p[i]])
            p_soft_con[i] = (kappa_delta_p[i] > 0)

        tmp += p == min([i for i in range(max_conj_size) if (kappa_delta_p[i] < len(delta_p[i]))])

        objective = sum([c for c in delta])  # get the amount of satisfied constraints from B

        # at least 1 violated and at least 1 satisfied
        # we want this to assure that each answer of the user will reduce
        # the set of candidates
        tmp += objective < len(delta)
        tmp += objective > 0

        # Try first without objective
        s = cp.SolverLookup.get("ortools", tmp)

        # run with the objective
        s.minimize(100 * p - p_soft_con[p])

        flag = s.solve(time_limit=self.time_limit)

        return flag
