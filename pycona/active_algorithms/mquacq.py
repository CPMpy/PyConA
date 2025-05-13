import time

from .algorithm_core import AlgorithmCAInteractive
from ..utils import get_kappa
from ..ca_environment.active_ca import ActiveCAEnv
from ..problem_instance import ProblemInstance
from ..answering_queries import Oracle, UserOracle
from .. import Metrics


class MQuAcq(AlgorithmCAInteractive):
    """
    MQuAcq is an implementation of the ICA_Algorithm that uses a modified QuAcq algorithm to learn constraints.
    """

    def __init__(self, ca_env: ActiveCAEnv = None):
        """
        Initialize the MQuAcq algorithm with an optional constraint acquisition environment.

        :param ca_env: An instance of ActiveCAEnv, default is None.
        """
        super().__init__(ca_env)

    def learn(self, instance: ProblemInstance, oracle: Oracle = UserOracle(), verbose=0, X=None, metrics: Metrics = None):
        """
        Learn constraints using the modified QuAcq algorithm by generating queries and analyzing the results.

        :param instance: the problem instance to acquire the constraints for
        :param oracle: An instance of Oracle, default is to use the user as the oracle.
        :param verbose: Verbosity level, default is 0.
        :param metrics: statistics logger during learning
        :param X: The set of variables to consider, default is None.
        :return: the learned instance
        """
        if X is None:
            X = instance.X
        assert isinstance(X, list), "When using .learn(), set parameter X must be a list of variables. Instead got: {}".format(X)
        assert set(X).issubset(set(instance.X)), "When using .learn(), set parameter X must be a subset of the problem instance variables. Instead got: {}".format(X)

        self.env.init_state(instance, oracle, verbose, metrics)

        if len(self.env.instance.bias) == 0:
            self.env.instance.construct_bias(X)

        while True:
            if self.env.verbose >= 3:
                print("Size of CL: ", len(self.env.instance.cl))
                print("Size of B: ", len(self.env.instance.bias))
                print("Number of queries: ", self.env.metrics.total_queries)
                print("MQuAcq-2 Queries: ", self.env.metrics.top_lvl_queries)
                print("FindScope Queries: ", self.env.metrics.findscope_queries)
                print("FindC Queries: ", self.env.metrics.findc_queries)

            # generate e in D^X accepted by C_l and rejected by B
            gen_start = time.time()
            Y = self.env.run_query_generation(X)
            gen_end = time.time()

            if len(Y) == 0:
                # if no query can be generated it means we have (prematurely) converged to the target network -----
                self.env.metrics.finalize_statistics()
                if self.env.verbose >= 1:
                    print(f"\nLearned {self.env.metrics.cl} constraints in "
                          f"{self.env.metrics.membership_queries_count} queries.")
                self.env.instance.bias = []
                return self.env.instance

            self.env.metrics.increase_generation_time(gen_end - gen_start)
            self.env.metrics.increase_generated_queries()
            self.find_all_cons(list(Y), set())

    def find_all_cons(self, Y, Scopes):
        """
        Recursively find all constraints that can be learned from the given query.

        :param Y: The query to be analyzed.
        :param Scopes: The set of scopes to be considered.
        :return: The set of learned scopes.
        """
        kappa = get_kappa(self.env.instance.bias, Y)
        if len(kappa) == 0:
            return set()

        NScopes = set()

        if len(Scopes) > 0:
            s = Scopes.pop()
            for x in s:
                Y2 = set(Y.copy())
                if x in Y2:
                    Y2.remove(x)

                scopes = self.find_all_cons(list(Y2), NScopes.union(Scopes))
                NScopes = NScopes.union(scopes)

        else:
            self.env.metrics.increase_top_queries()
            if self.env.ask_membership_query(Y):
                self.env.remove_from_bias(kappa)
            else:
                scope = self.env.run_find_scope(Y)
                c = self.env.run_findc(scope)
                self.env.add_to_cl(c)

                NScopes.add(frozenset(scope))

                NScopes = NScopes.union(self.find_all_cons(Y, NScopes.copy()))

        return NScopes
