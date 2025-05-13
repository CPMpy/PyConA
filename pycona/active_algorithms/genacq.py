import time
from itertools import product

from .algorithm_core import AlgorithmCAInteractive
from ..ca_environment.active_ca import ActiveCAEnv
from ..utils import get_relation, get_scope, get_kappa, replace_variables
from ..problem_instance import ProblemInstance
from ..answering_queries import Oracle, UserOracle
from .. import Metrics


class GenAcq(AlgorithmCAInteractive):

    """
    GenAcq algorithm, using generalization queries on given types of variables. From:

    "Boosting Constraint Acquisition with Generalization Queries", ECAI 2014.
    """

    def __init__(self, ca_env: ActiveCAEnv = None, types=None, qg_max=3):
        """
        Initialize the GenAcq algorithm with an optional constraint acquisition environment.

        :param ca_env: An instance of ActiveCAEnv, default is None.
        : param types: list of types of variables given by the user
        : param qg_max: maximum number of generalization queries
        """
        super().__init__(ca_env)
        self._negativeQ = []
        self._qg_max = qg_max
        self._types = types if types is not None else []

    def learn(self, instance: ProblemInstance, oracle: Oracle = UserOracle(), verbose=0, metrics: Metrics = None, X=None):
        """
        Learn constraints using the GenAcq algorithm by generating queries and analyzing the results.
        Using generalization queries on given types of variables.

        :param instance: the problem instance to acquire the constraints for
        :param oracle: An instance of Oracle, default is to use the user as the oracle.
        :param verbose: Verbosity level, default is 0.
        :param metrics: statistics logger during learning
        :return: the learned instance
        """
        self.env.init_state(instance, oracle, verbose, metrics)

        if X is None:
            X = list(self.env.instance.variables.flat)

        if len(self.env.instance.bias) == 0:
            self.env.instance.construct_bias(X)

        while True:
            if self.env.verbose > 0:
                print("Size of CL: ", len(self.env.instance.cl))
                print("Size of B: ", len(self.env.instance.bias))
                print("Number of Queries: ", self.env.metrics.total_queries)
                print("Number of Generalization Queries: ", self.env.metrics.generalization_queries_count)
                print("Number of Membership Queries: ", self.env.metrics.membership_queries_count)


            gen_start = time.time()
            Y = self.env.run_query_generation(X)
            gen_end = time.time()

            if len(Y) == 0:
                # if no query can be generated it means we have (prematurely) converged to the target network -----
                self.env.metrics.finalize_statistics()
                if self.env.verbose >= 1:
                    print(f"\nLearned {self.env.metrics.cl} constraints in "
                          f"{self.env.metrics.total_queries} queries.")
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
                self.generalize(get_relation(c, self.env.instance.language),c)



    def generalize(self, r, c):
        """
        Generalize function presented in
        "Boosting Constraint Acquisition with Generalization Queries", ECAI 2014.


        :param r: The index of a relation in gamma.
        :param c: The constraint to generalize.
        :return: List of learned constraints.
        """
        # Get the scope variables of constraint c
        scope_vars = get_scope(c)
        
        # Find all possible type sequences for the variables in the scope
        type_sequences = []
        for var in scope_vars:
            var_types = []
            for type_group in self._types:
                if var.name in type_group:
                    var_types.append(type_group)
            type_sequences.append(var_types)

        # Generate all possible combinations of type sequences
        all_type_sequences = list(product(*type_sequences))
        
        # Filter out sequences based on NegativeQ and NonTarget
        filtered_sequences = []
        for s in all_type_sequences:

            # Check if any negative sequence is a subset of current sequence
            if s in self._negativeQ:
                continue

            # Check if any non-target constraint has same relation and vars in sequence
            if any(get_relation(c2, self.env.instance.language) == r and 
                   all(any(var in set(type_group) for type_group in s) for var in get_scope(c2))
                   for c2 in set(self.env.instance.excluded_cons)):
                continue

            filtered_sequences.append(s)
        
        all_type_sequences = filtered_sequences
        
        gq_counter = 0

        # Sort sequences by number of distinct elements (ascending)
        all_type_sequences.sort(key=lambda seq: len(set().union(*seq)))

        while len(all_type_sequences) > 0 and gq_counter < self._qg_max:
            Y = all_type_sequences.pop(0)    
            
            # Instead of getting constraints from bias, generate them for this type sequence
            B = []
            
            # Generate all possible variable combinations
            var_combinations = list(product(*Y))
            # Create constraints for each variable combination
            for var_comb in var_combinations:

                if len(set(var_comb)) != len(var_comb):  # No duplicates
                    continue
                # Sort var_comb based on variable names 
                var_comb = sorted(var_comb, key=lambda var: var.name)
                
                abs_vars = get_scope(self.env.instance.language[r])
                replace_dict = dict()
                for i, v in enumerate(var_comb):
                    replace_dict[abs_vars[i]] = v
                constraint = replace_variables(self.env.instance.language[r], replace_dict)
                
                # Skip already learned or excluded constraints
                if constraint not in set(self.env.instance.cl) and constraint not in set(self.env.instance.excluded_cons):
                    B.append(constraint)

            # If generalization query is accepted
            if self.env.ask_generalization_query(self.env.instance.language[r], B):
                self.env.add_to_cl(B)
                gq_counter = 0
            else:
                gq_counter += 1
                self._negativeQ.append(Y)

        