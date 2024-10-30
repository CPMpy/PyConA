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

    def __init__(self, ca_system: ActiveCAEnv = None, time_limit=0.2, findscope=None):
        """
        Initialize the FindC2 class.

        :param ca_system: The constraint acquisition system.
        :param time_limit: The time limit for findc query generation.
        :param findscope: The function to find the scope.
        """
        super().__init__(ca_system, time_limit)
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

                scope2 = self.ca.run_find_scope(list(scope))

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

        # example needs to satisfy learned constrains
        constraints = L

        # example needs to violate at least one from Delta but not all
        delta_sat = cp.sum([c for c in delta])  # Get the amount of satisfied constraints from Delta
        constraints.append(delta_sat > 0)
        constraints.append(delta_sat < len(delta))

        max_conj_size = get_max_conjunction_size(delta)
        delta_p = get_delta_p(delta)

        p = 0  # conjunction size to focus on

        while p < max_conj_size:

            tmp_model = cp.Model(constraints)
            delta_p_sat = cp.sum([c for c in delta_p[p]])  # Get the amount of satisfied constraints from Delta_p
            tmp_model += delta_p_sat > 0  # example needs to not violate all from Delta_p

            # but also violate at least one if possible
            b = cp.boolvar()
            tmp_model += b.implies(delta_p_sat < len(delta_p[p]))
            tmp_model.maximize(b)

            if tmp_model.solve():
                return True  # query found, return true

            p += 1  # increase p if no query found for current conjunction size

        return False  # no query could be found
