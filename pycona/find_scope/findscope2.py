from ..ca_environment.active_ca import ActiveCAEnv
from .findscope_core import FindScopeBase
from ..utils import get_kappa, get_con_subset
from ..find_scope.findscope_obj import split_proba

class FindScope2(FindScopeBase):
    """
    This is the version of the FindScope function that was presented in
    Bessiere, Christian, et al., "Learning constraints through partial queries", AIJ 2023
    """

    def __init__(self, ca_env: ActiveCAEnv = None, split_func=split_proba, time_limit=0.2):
        """
        Initialize the FindScope2 class.

        :param ca_env: The constraint acquisition environment.
        :param time_limit: The time limit for findscope query generation.
        """
        super().__init__(ca_env, time_limit, split_func=split_func)
        self._kappaB = []

    def run(self, Y, kappa=None):
        """
        Run the FindScope2 algorithm.

        :param Y: A set of variables.
        :return: The scope of the partial example.
        :raises Exception: If the partial example is not a negative example.
        """
        assert self.ca is not None

        kappaB = kappa if kappa is not None else get_kappa(self.ca.instance.bias, Y)
        self._kappaB = kappaB
        if len(self._kappaB) == 0:
            raise Exception(f"The partial example e_Y, on the subset of variables Y given in FindScope, "
                            f"must be a negative example")
        scope = self._find_scope(set(), Y)

        return scope

    def _find_scope(self, R, Y):
        """
        Find the scope of the partial example.

        :param R: A set of variables.
        :param Y: A set of variables.
        :return: The scope of the partial example.
        :raises Exception: If kappaB is not part of the bias.
        """

        RY = R.union(Y)
        kappaBRY = get_con_subset(self._kappaB, list(RY))

        # if ask(e_R) = yes: B \setminus K(e_R)
        # need to project 'e' down to vars in R,
        # will show '0' for None/na/" ", should create object nparray instead
        kappaBR = get_con_subset(kappaBRY, list(R))
        if len(kappaBR) > 0:

            self.ca.metrics.increase_findscope_queries()
            if self.ca.ask_membership_query(R):
                self.ca.remove_from_bias(kappaBR)
                self._kappaB = list(set(self._kappaB) - set(kappaBR))
                kappaBRY = get_con_subset(self._kappaB, list(RY))

            else:
                return set()

        if len(Y) == 1:
            return set(Y)


        # Create Y1, Y2 -------------------------
        proba = self.ca.bias_proba if hasattr(self.ca, 'bias_proba') else []
        Y1, Y2 = self.split_func(Y=Y, R=R, kappaB=kappaBRY, P_c=proba)

        S1 = set()
        S2 = set()

        # R U Y1
        RY1 = R.union(Y1)
        kappaBRY1 = get_con_subset(kappaBRY, RY1)

        if len(kappaBRY1) < len(kappaBRY):
            S1 = self._find_scope(RY1, Y2)
            kappaBRY = get_con_subset(self._kappaB, list(RY))  # update in case constraints were removed

        # R U S1
        RS1 = R.union(S1)

        kappaBRS1 = get_con_subset(kappaBRY, RS1)

        if len(kappaBRS1) < len(kappaBRY):
            S2 = self._find_scope(RS1, Y1)

        return S1.union(S2)
