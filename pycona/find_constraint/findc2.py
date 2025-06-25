import cpmpy as cp
import copy

from ..ca_environment.active_ca import ActiveCAEnv
from .utils import get_max_conjunction_size, get_delta_p
from .findc_core import FindCBase
from .utils import join_con_net
from ..utils import restore_scope_values, get_con_subset, check_value, get_scope


class FindC2(FindCBase):
    """
    This is the version of the FindC function that was presented in
    Bessiere, Christian, et al., "Learning constraints through partial queries", AIJ 2023

    This function works also for non-normalised target networks!
    """

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

        scope_values = [x.value() for x in scope]
        
        # Initialize delta
        delta = get_con_subset(self.ca.instance.bias, scope)
        kappaD = [c for c in delta if check_value(c) is False]
        delta = join_con_net(delta, kappaD)

        # We need to take into account only the constraints in the scope we search on
        sub_cl = get_con_subset(self.ca.instance.cl, scope)

        while True:

            # Try to generate a counter example to reduce the candidates
            if self.generate_findc_query(sub_cl, delta) is False:

                # If no example could be generated
                # Check if delta is the empty set, and if yes then collapse
                if len(delta) == 0:
                    raise Exception("Collapse, the constraint we seek is not in B")

                restore_scope_values(scope, scope_values)

                # Return random c in delta otherwise (if more than one, they are equivalent w.r.t. C_l)
                # Choose the constraint with the smallest number of conjunctions
                delta = sorted(delta, key=lambda x: len(x.args))
                return delta[0]

            self.ca.metrics.increase_findc_queries()

            if self.ca.ask_membership_query(scope):
                # delta <- delta \setminus K_{delta}(e)
                delta = [c for c in delta if check_value(c) is not False]
                [self.ca.remove_from_bias(c) for c in delta if check_value(c) is False]

            else:  # user says UNSAT
                # delta <- joint(delta,K_{delta}(e))

                kappaD = [c for c in delta if check_value(c) is False]

                #scope2 = self.ca.run_find_scope(list(scope), kappaD)  # TODO: replace with real findscope arguments when done!

                #if len(scope2) < len(scope):
                #    self.run(scope2)
                #else:
                delta = join_con_net(delta, kappaD)

    def generate_findc_query(self, L, delta):
        """
        Changes directly the values of the variables

        :param L: learned network in the given scope
        :param delta: candidate constraints in the given scope
        :return: Boolean value representing a success or failure on the generation
        """

        tmp = cp.Model(L)        

        satisfied_delta = sum([c for c in delta])  # get the amount of satisfied constraints from B

        scope = get_scope(delta[0])
        # at least 1 violated and at least 1 satisfied
        # we want this to assure that each answer of the user will reduce
        # the set of candidates
        tmp += satisfied_delta < len(delta)
        tmp += satisfied_delta > 0


        max_conj_size = get_max_conjunction_size(delta)
        delta_p = get_delta_p(delta)

        for p in range(max_conj_size):
            s = cp.SolverLookup.get("ortools", tmp)

            kappa_delta_p = sum([c for c in delta_p[p]])
            s += kappa_delta_p < len(delta_p[p])
            

            if not s.solve(): # if a solution is found
                continue

            # Next solve will change the values of the variables in lY
            # so we need to return them to the original ones to continue if we don't find a solution next
            values = [x.value() for x in scope]


            p_soft_con = (kappa_delta_p > 0)
            
            # run with the objective
            s.maximize(p_soft_con)

            # So a solution was found, try to find a better one now
            s.solution_hint(scope, values)

            flag = s.solve(time_limit=self.time_limit, num_workers=8)
            if not flag:
                restore_scope_values(scope, values)
            return True

        return False