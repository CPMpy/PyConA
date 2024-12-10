from ..ca_environment.active_ca import ActiveCAEnv
from .findscope_core import FindScopeBase
from ..utils import get_kappa


class FindScope(FindScopeBase):
    """
    This is the version of the FindScope function that was presented in
    Bessiere, Christian, et al., "Constraint acquisition via Partial Queries", IJCAI 2013.
    """

    def __init__(self, ca_env: ActiveCAEnv = None, time_limit=0.2):
        """
        Initialize the FindScope class.

        :param ca_env: The constraint acquisition environment.
        :param time_limit: The time limit for findscope query generation.
        """
        super().__init__(ca_env, time_limit)

    def run(self, Y):
        """
        Run the FindScope algorithm.

        :param Y: A set of variables.
        :return: The scope of the partial example.
        """
        assert self.ca is not None
        scope = self._find_scope(set(), Y, do_ask=False)
        return scope

    def _find_scope(self, R, Y, do_ask):
        """
        Find the scope of a violated constraint.

        :param R: A set of variables.
        :param Y: A set of variables.
        :param do_ask: A boolean indicating whether to ask a membership query.
        :return: The scope of the partial example.
        """
        if do_ask:
            # if ask(e_R) = yes: B \setminus K(e_R)
            # need to project 'e' down to vars in R,
            # will show '0' for None/na/" ", should create object nparray instead

            self.ca.metrics.increase_findscope_queries()

            if self.ca.ask_membership_query(R):
                kappaB = get_kappa(self.ca.instance.bias, R)
                self.ca.remove_from_bias(kappaB)
            else:
                return set()

        if len(Y) == 1:
            return set(Y)

        s = len(Y) // 2
        Y1, Y2 = Y[:s], Y[s:]

        S1 = self._find_scope(R.union(Y1), Y2, True)
        S2 = self._find_scope(R.union(S1), Y1, len(S1) > 0)

        return S1.union(S2)

