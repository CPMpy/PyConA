import cpmpy as cp
import copy

from ..ca_environment.active_ca import ActiveCAEnv
from .utils import get_max_conjunction_size, get_delta_p, join_con_net, unravel_conjunctions
from .findc_core import FindCBase
from ..utils import restore_scope_values, get_con_subset, check_value, get_scope


class FindC2(FindCBase):
    """
    Implementation of the FindC algorithm from Bessiere et al., "Learning constraints through partial queries" (AIJ 2023).
    
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

        Returns:
            callable: The function used to determine constraint scopes
        """
        return self._findscope

    @findscope.setter
    def findscope(self, findscope):
        """
        Set the findscope function to be used.

        Args:
            findscope (callable): The function to be used for determining constraint scopes
        """
        self._findscope = findscope

    def run(self, scope):
        """
        Execute the FindC2 algorithm to learn constraints within a given scope.

        Args:
            scope (list): Variables defining the scope in which to search for constraints

        Returns:
            list: The constraint(s) found in the given scope.

        Raises:
            Exception: If the target constraint is not in the bias (search space).
        """
        assert self.ca is not None
        scope_values = [x.value() for x in scope]
        
        # Initialize delta with constraints from bias that match the scope
        delta = get_con_subset(self.ca.instance.bias, scope)
        delta = [c for c in delta if len(get_scope(c)) == len(scope)]

        # Join the constraints in delta with the violated constraints in kappaD
        kappaD = [c for c in delta if check_value(c) is False]
        delta = join_con_net(delta, kappaD)

        # Get subset of learned constraints in the current scope
        sub_cl = get_con_subset(self.ca.instance.cl, scope)

        while True:
            # Generate a query to distinguish between candidate constraints
            if self.generate_findc_query(sub_cl, delta) is False:
                # If no example could be generated
                # Check if delta is the empty set, and if yes then collapse
                if len(delta) == 0:
                    raise Exception("Collapse, the constraint we seek is not in B")

                restore_scope_values(scope, scope_values)

                # Unravel nested AND constraints
                delta_unraveled = unravel_conjunctions(delta)
                
                # Return the smallest equivalent conjunction (if more than one, they are equivalent w.r.t. C_l)
                delta_unraveled = sorted(delta_unraveled, key=lambda x: len(x))
                return delta_unraveled[0]

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
                    # Recursively learn constraint in sub-scope
                    c = self.run(scope2)
                    self.ca.add_to_cl(c)
                    sub_cl.append(c)
                else:
                    delta = join_con_net(delta, kappaD)

    def generate_findc_query(self, L, delta):
        """
        Generate a query that helps distinguish between candidate constraints.

        Args:
            L (list): Currently learned constraints in the scope
            delta (list): Candidate constraints to distinguish between

        Returns:
            bool: True if a query was generated successfully, False otherwise

        Note:
            The method directly modifies variable values in the constraint network
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
            
            # Solve without objective for start
            if not s.solve(): # if a solution is not found
                continue

            # Next solve will change the values of the variables in lY
            # so we need to return them to the original ones to continue if we don't find a solution next
            values = [x.value() for x in scope]

            p_soft_con = (kappa_delta_p > 0)
            
            # So a solution was found, try to find a better one now
            # set the objective
            s.maximize(p_soft_con)

            # Give hint with previous solution to the solver
            s.solution_hint(scope, values)

            # Solve with objective
            flag = s.solve(time_limit=self.time_limit, num_workers=8)
            if not flag:
                restore_scope_values(scope, values)
            return True

        return False