import copy

from .mquacq2 import MQuAcq2
from .algorithm_core import AlgorithmCAInteractive
from ..ca_environment.active_ca import ActiveCAEnv
from ..problem_instance import ProblemInstance
from ..answering_queries import Oracle, UserOracle
from .. import Metrics
from ..ca_environment import ProbaActiveCAEnv


class GrowAcq(AlgorithmCAInteractive):
    """
    GrowAcq ICA algorithm from:
    Dimos Tsouros, Senne Berden, and Tias Guns. "Guided Bottom-Up Interactive Constraint Acquisition." CP, 2023
    """

    def __init__(self, ca_env: ActiveCAEnv = None, inner_algorithm: AlgorithmCAInteractive = None):
        """
        Initialize the GrowAcq algorithm with an optional constraint acquisition system and inner algorithm.

        :param ca_env: An instance of CASystem, default is None.
        :param inner_algorithm: An instance of ICA_Algorithm to be used as the inner algorithm, default is MQuAcq2.
        """
        env = ca_env if ca_env is not None else ProbaActiveCAEnv()
        super().__init__(env)
        self.inner_algorithm = inner_algorithm if inner_algorithm is not None else MQuAcq2(ca_env)

    def learn(self, instance: ProblemInstance, oracle: Oracle = UserOracle(), verbose=0, metrics: Metrics = None):
        """
        Learn constraints by incrementally adding variables and using the inner algorithm to learn constraints
        for each added variable.

        :param instance: the problem instance to acquire the constraints for
        :param oracle: An instance of Oracle, default is to use the user as the oracle.
        :param verbose: Verbosity level, default is 0.
        :param metrics: statistics logger during learning
        :return: the learned instance
        """
        self.env.init_state(instance, oracle, verbose, metrics)

        if verbose >= 1:
            print(f"Running growacq with {self.inner_algorithm} as inner algorithm")

        self.inner_algorithm.env = copy.copy(self.env)

        self.env.instance.X = []

        n_vars = len(self.env.instance.variables.flat)
        for x in self.env.instance.variables.flat:
            # we 'grow' the inner bias by adding one extra variable at a time
            self.env.instance.X.append(x)
            # add the constraints involving x and other added variables
            self.env.instance.construct_bias_for_var(x)
            if verbose >= 3:
                print(f"Added variable {x} in GrowAcq")
                print("size of B in growacq: ", len(self.env.instance.bias))

            if verbose >= 2:
                print(f"\nGrowAcq: calling inner_algorithm for {len(self.env.instance.X)}/{n_vars} variables")
            self.env.instance = self.inner_algorithm.learn(self.env.instance, oracle, verbose=verbose, metrics=self.env.metrics)

            if verbose >= 3:
                print("C_L: ", len(self.env.instance.cl))
                print("B: ", len(self.env.instance.bias))
                print("Number of queries: ", self.env.metrics.membership_queries_count)
                print("Top level Queries: ", self.env.metrics.top_lvl_queries)
                print("FindScope Queries: ", self.env.metrics.findscope_queries)
                print("FindC Queries: ", self.env.metrics.findc_queries)
        # end of for loop, full bias visited

        if verbose >= 3:
            print("Number of queries: ", self.env.metrics.membership_queries_count)
            print("Number of recommendation queries: ", self.env.metrics.recommendation_queries_count)
            print("Number of generalization queries: ", self.env.metrics.generalization_queries_count)
            print("Top level Queries: ", self.env.metrics.top_lvl_queries)
            print("FindScope Queries: ", self.env.metrics.findscope_queries)
            print("FindC Queries: ", self.env.metrics.findc_queries)

        self.env.metrics.finalize_statistics()
        return self.env.instance
