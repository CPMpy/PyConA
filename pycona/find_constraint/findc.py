from ..ca_environment.active_ca import ActiveCAEnv
from .findc_core import FindCBase
from ..utils import restore_scope_values, get_con_subset, check_value, get_kappa
from .findc_obj import findc_obj_splithalf


class FindC(FindCBase):
    """
    This is the version of the FindC function that was presented in
    Bessiere, Christian, et al., "Constraint acquisition via Partial Queries", IJCAI 2013.

    This version works only for normalised target networks!
    A modification that can also learn conjunction of constraints in each scope is described in the article:
    Bessiere, Christian, et al., "Learning constraints through partial queries", AIJ 2023
    """

    def __init__(self, ca_env: ActiveCAEnv = None, time_limit=0.2, *, findc_obj=findc_obj_splithalf):
        """
        Initialize the FindC class.

        :param ca_env: The constraint acquisition environment.
        :param time_limit: The time limit for findc query generation.
        :param findc_obj: The function to use for findc objective, default is findc_obj_splithalf.
        """
        super().__init__(ca_env, time_limit, findc_obj=findc_obj)

    def run(self, scope):
        """
        Run the FindC algorithm.

        :param scope: The scope in which we search for a constraint.
        :return: The constraint found.
        """
        assert self.ca is not None

        # Initialize delta
        delta = get_con_subset(self.ca.instance.bias, scope)
        delta = [c for c in delta if check_value(c) is False]

        if len(delta) == 1:
            c = delta[0]
            return c

        if len(delta) == 0:
            raise Exception(f"Collapse, the constraint we seek is not in B: {get_kappa(self.ca.oracle.constraints,scope)}")
        # We need to take into account only the constraints in the scope we search on
        sub_cl = get_con_subset(self.ca.instance.cl, scope)

        # Save current variable values
        scope_values = [x.value() for x in scope]

        while True:

            # Generate a counter example to reduce the candidates
            flag = self.generate_findc_query(sub_cl, delta)

            if flag is False:
                # If no example could be generated
                # Check if delta is the empty set, and if yes then collapse
                if len(delta) == 0:
                    raise Exception("Collapse, the constraint we seek is not in B: ")

                restore_scope_values(scope, scope_values)

                # Return random c in delta otherwise (if more than one, they are equivalent w.r.t. C_l)
                return delta[0]

            self.ca.metrics.increase_findc_queries()

            if self.ca.ask_membership_query(scope):
                # delta <- delta \setminus K_{delta}(e)
                [self.ca.remove_from_bias(c) for c in delta if check_value(c) is False]
                delta = [c for c in delta if check_value(c) is not False]
            else:  # user says UNSAT
                # delta <- K_{delta}(e)
                [self.ca.remove_from_bias(c) for c in delta if check_value(c) is not False]
                delta = [c for c in delta if check_value(c) is False]
