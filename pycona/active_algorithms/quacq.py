import time

from .algorithm_core import AlgorithmCAInteractive
from ..problem_instance import ProblemInstance
from ..answering_queries import Oracle, UserOracle
from ..ca_environment.active_ca import ActiveCAEnv
from ..utils import get_kappa
from .. import Metrics


class QuAcq(AlgorithmCAInteractive):
    """
    QuAcq is an implementation of the ICA_Algorithm that uses the QuAcq algorithm to learn constraints.
    """

    def __init__(self, ca_env: ActiveCAEnv = None):
        """
        Initialize the QuAcq algorithm with an optional constraint acquisition environment.

        :param ca_env: An instance of ActiveCAEnv, default is None.
        """
        super().__init__(ca_env)

    def learn(self, instance: ProblemInstance, oracle: Oracle = UserOracle(), verbose=0, X=None, metrics: Metrics = None):
        """
        Learn constraints using the QuAcq algorithm by generating queries and analyzing the results.

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
            if self.env.verbose > 2:
                print("Size of CL: ", len(self.env.instance.cl))
                print("Size of B: ", len(self.env.instance.bias))
                print("Number of Queries: ", self.env.metrics.membership_queries_count)

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
            self.env.metrics.increase_top_queries()
            kappaB = get_kappa(self.env.instance.bias, Y)

            answer = self.env.ask_membership_query(Y)
            if answer:
                # it is a solution, so all candidates violated must go
                # B <- B \setminus K_B(e)
                self.env.remove_from_bias(kappaB)

            else:  # user says UNSAT

                scope = self.env.run_find_scope(Y)
                c = self.env.run_findc(scope)
                self.env.add_to_cl(c)


