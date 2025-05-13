import time

from cpmpy.transformations.get_variables import get_variables

from .algorithm_core import AlgorithmCAInteractive
from ..ca_environment.active_ca import ActiveCAEnv
from ..utils import get_relation, get_scope, get_kappa
from ..problem_instance import ProblemInstance
from ..answering_queries import Oracle, UserOracle
from .. import Metrics


class PQuAcq(AlgorithmCAInteractive):

    """
    PQuAcq is a variation of QuAcq, using Predict&Ask function and recommendation queries. Presented in
    "Constraint Acquisition with Recommendation Queries", IJCAI 2016.
    """

    def __init__(self, ca_env: ActiveCAEnv = None):
        """
        Initialize the PQuAcq algorithm with an optional constraint acquisition environment.

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
            if self.env.verbose > 0:
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
                self.predictAsk(get_relation(c, self.env.instance.language))


    def predictAsk(self, r):
        """
        Predict&Ask function presented in "Constraint Acquisition with Recommendation Queries", IJCAI 2016.

        :param r: The index of a relation in gamma.
        :return: List of learned constraints.
        """
        try:
            import networkx as nx
        except ImportError:
            raise ImportError("To use the predictAsk function of PQuAcq, networkx needs to be installed")

        assert isinstance(r, int) and r in range(len(self.env.instance.language)), \
            "predictAsk input must be the index of a relation in the language"

        alpha = 4  # \alpha from the paper, set to 4
        L = []
        C = [c for c in self.env.instance.cl if get_relation(c, self.env.instance.language) == r]

        # Project Y to those in C that have relation r
        Y = [v.name for v in get_variables(C)]
        E = [tuple([v.name for v in get_scope(c)]) for c in C]  # all scopes

        # Create the graph
        G = nx.Graph()
        G.add_nodes_from(Y)
        G.add_edges_from(E)

        B = [c for c in self.env.instance.bias if get_relation(c, self.env.instance.language) == r and
             frozenset(get_scope(c)).issubset(frozenset(Y))]
        D = [tuple([v.name for v in get_scope(c)]) for c in B]  # missing edges that can be completed (exist in B)
        neg = 0  # counter of negative answers

        while len(D) > 0 and neg < alpha:  # alpha is the cutoff

            scores = list(nx.adamic_adar_index(G, D))

            # Find the index of the tuple with the maximum score
            score_values = [score for _, _, score in scores]
            max_index = score_values.index(max(score_values))
            c = B[max_index]
            B.pop(max_index)
            D.pop(max_index)

            if self.env.ask_recommendation_query(c):
                self.env.add_to_cl(c)
                E.append(tuple([v.name for v in get_scope(c)]))
                G.add_edges_from(E)
                neg = 0
            else:
                neg += 1
                self.env.remove_from_bias(c)
