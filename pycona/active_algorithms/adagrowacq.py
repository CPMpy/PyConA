import cpmpy as cp
from .quacq import QuAcq
from .algorithm_core import AlgorithmCAInteractive
from ..ca_environment.active_ca import ActiveCAEnv
from ..problem_instance import ProblemInstance
from ..answering_queries import Oracle, UserOracle
from .. import Metrics
from ..ca_environment import ProbaActiveCAEnv
import random
from scipy.stats import entropy
from ..utils import get_scope, get_con_subset, get_var_dims
import numpy as np


class AdaGrowAcq(AlgorithmCAInteractive):
    """
    GrowAcq ICA algorithm from:
    Dimos Tsouros, Senne Berden, and Tias Guns. "Guided Bottom-Up Interactive Constraint Acquisition." CP, 2023
    """

    def __init__(self, ca_env: ActiveCAEnv = None, inner_algorithm: AlgorithmCAInteractive = None, adaptive_grow: int = 3, choose_variables: str = "random"):
        """
        Initialize the AdaGrowAcq algorithm with an optional constraint acquisition environment and inner algorithm.

        :param ca_env: An instance of ActiveCAEnv, default is None.
        :param inner_algorithm: An instance of ICA_Algorithm to be used as the inner algorithm, default is MQuAcq2.
        :param adaptive_grow: The strategy to use for adaptive grow, default is 3.
        :param choose_variables: The strategy to choose variables, default is "random".
        """
        env = ca_env if ca_env is not None else ProbaActiveCAEnv()
        super().__init__(env)
        self.inner_algorithm = inner_algorithm if inner_algorithm is not None else QuAcq(ca_env)
        self._adaptive_grow = adaptive_grow    
        self._choose_variables = choose_variables

    def learn(self, instance: ProblemInstance, oracle: Oracle = UserOracle(), verbose=0, metrics: Metrics = None, X=None):
        """
        Learn constraints by incrementally adding variables and using the inner algorithm to learn constraints
        for each added variable.

        :param instance: the problem instance to acquire the constraints for
        :param oracle: An instance of Oracle, default is to use the user as the oracle.
        :param verbose: Verbosity level, default is 0.
        :param metrics: statistics logger during learning
        :param X: list of variables to consider, default is all variables
        :return: the learned instance
        """
        self.env.init_state(instance, oracle, verbose, metrics)

        if verbose >= 1:
            print(f"Running growacq with {self.inner_algorithm} as inner algorithm")

        self.inner_algorithm.env = self.env

        #self.env.instance.construct_bias() # should construct bias for all variables
        
        n_vars = len(self.env.instance.X)

        if X is None:
            X = list(self.env.instance.X) # X is the set of variables that have not been added to the model        

        Y = [] # Y is the set of variables that have been added to the model

        v = 1
        g = 1
        it = 0

        init_bias = list(self.env.instance.bias)
        init_bias_provided = len(init_bias) > 0
        
        while len(Y) < n_vars:
            it += 1

            Y_new = self.choose_variables(X, Y, v)
            if verbose >= 3:
                print(f"Added to GrowAcq: {set(Y_new) - set(Y)}")

            # add the constraints involving x and other added variables
            if init_bias_provided:
                visible_now = set(get_con_subset(init_bias, Y_new))
                self.env.instance.bias = list(visible_now)
                init_bias = set(init_bias) - visible_now
            else:
                self.env.instance.construct_bias_for_vars(set(Y_new) - set(Y), Y)
                if verbose >= 3:
                    print(f"Created {len(self.env.instance.bias)} constraints")

            Y = Y_new

            if verbose >= 0:
                print(f"\nGrowAcq: calling inner_algorithm for {len(Y)}/{n_vars} variables")
            cl_size = len(self.env.instance.cl)
            self.env.instance = self.inner_algorithm.learn(self.env.instance, oracle, verbose=verbose, metrics=self.env.metrics, X=Y)
            self.env.metrics.print_statistics()
            if len(self.env.instance.cl) == cl_size:
                if self._adaptive_grow == 1:   
                    v += v
                elif self._adaptive_grow == 2  or self._adaptive_grow == 4:
                    v = len(Y)
                elif self._adaptive_grow == 3 or self._adaptive_grow == 5:
                    v = int((it/g)*len(Y))
            else:
                g += 1
                if self._adaptive_grow == 3:
                    v = int((it/g))
                elif self._adaptive_grow == 4:
                    v = len(Y)
                else:
                    v = 1

            v = min(v, len(X) - len(Y))

            if verbose >= 3:
                print("C_L: ", len(self.env.instance.cl))
                print("B: ", len(self.env.instance.bias))
                print("Number of queries: ", self.env.metrics.total_queries)
                print("Top level Queries: ", self.env.metrics.top_lvl_queries)
                print("FindScope Queries: ", self.env.metrics.findscope_queries)
                print("FindC Queries: ", self.env.metrics.findc_queries)

        # end of for loop, all variables visited

        if verbose >= 3:
            print("Number of queries: ", self.env.metrics.membership_queries_count)
            print("Number of recommendation queries: ", self.env.metrics.recommendation_queries_count)
            print("Top level Queries: ", self.env.metrics.top_lvl_queries)
            print("FindScope Queries: ", self.env.metrics.findscope_queries)
            print("FindC Queries: ", self.env.metrics.findc_queries)

        self.env.metrics.finalize_statistics()
        return self.env.instance

    def get_inner_algorithm(self):
        return self.inner_algorithm

    def set_inner_algorithm(self, inner_algorithm):
        self.inner_algorithm = inner_algorithm

    def choose_variables(self, X, Y, v):
        """
        Choose a set of variables based on Manhattan distance from already selected variables.
        Variables that are furthest from Y in terms of tensor coordinates are selected first.
        """
        X_temp = list(set(X) - set(Y))
            
        if not Y:
            random.shuffle(X_temp)
            chosen_vars = [X_temp.pop() for _ in range(v)]  # Get a random variable
            chosen_vars.extend(Y)

        if self._choose_variables == "random":
            random.shuffle(X_temp)
            chosen_vars = [X_temp.pop() for _ in range(v)]  # Get a random variable
            chosen_vars.extend(Y)

        elif self._choose_variables == "last_neg_gen":

            if self.env._last_neg_gen is None:
                random.shuffle(X_temp)
                chosen_vars = [X_temp.pop() for _ in range(v)]  # Get a random variable
                chosen_vars.extend(Y)
                return chosen_vars

            chosen_vars = list(Y)

            # Train classifier if we have learned constraints
            if len(self.env.instance.cl) > 0:
                self.env._train_classifier()
            
            ccs = self.env._last_neg_gen.generate_ground_constraints(self.env.instance)
            # Create list of features for all constraints of the last negative generalization at once
            features_list = [self.env.feature_representation.featurize_constraint(c) for c in ccs]
            # Batch predict probabilities for all constraints
            probas = self.env.classifier_uncertanty.predict_proba(features_list)

            # Pre-calculate constraint entropies once
            constraint_entropy = {
                c: entropy([probas[i][1], 1-probas[i][1]], base=2)
                for i, c in enumerate(ccs)
            }

            # Cache related constraints for each variable
            var_constraints = {
                var: [c for c in ccs if var in set(get_scope(c))]
                for var in X_temp
            }

            # Select v variables with highest uncertainty scores
            for _ in range(min(v, len(X_temp))):
                # Calculate uncertainty scores for each variable in X_temp
                var_scores = {}
                for var in X_temp:
                    # Filter cached constraints to only those involving chosen variables
                    related_constraints = [
                        c for c in var_constraints[var]
                        if all(v in set(chosen_vars + [var]) for v in get_scope(c))
                    ]
                    
                    if related_constraints:
                        # Use pre-calculated entropies
                        var_scores[var] = max(constraint_entropy[c] for c in related_constraints)
                    else:
                        var_scores[var] = 0

                best_var = max(var_scores.items(), key=lambda x: x[1])[0]
                chosen_vars.append(best_var)
                X_temp = set(X_temp) - {best_var}
                del var_scores[best_var]
                del var_constraints[best_var]  # Clean up cache for selected variable.

        elif self._choose_variables == "min_manhattan":
            chosen_vars = list(Y)

            # Select v variables with highest distances (most different)
            for _ in range(min(v, len(X_temp))):

                # Calculate Manhattan distance scores for each variable in X_temp
                var_scores = {}
                for var in X_temp:
                    # Get the dimensions of the current variable   
                    var_dims = get_var_dims(var)

                    # Calculate maximum Manhattan distance to any variable in Y
                    min_distance = float('inf')
                    for y_var in chosen_vars:
                        # Get dimensions of the Y variable
                        y_dims = get_var_dims(y_var)

                        # Calculate Manhattan distance between var_dims and y_dims
                        distance = sum(abs(v1 - v2) for v1, v2 in zip(var_dims, y_dims))

                        # Keep track of minimum distance to any Y variable  
                        min_distance = min(min_distance, distance)
                    
                    # Store minimum Manhattan distance for this variable
                    var_scores[var] = min_distance

                best_var = min(var_scores.items(), key=lambda x: x[1])[0]
                chosen_vars.append(best_var)
                X_temp = set(X_temp) - {best_var}
                del var_scores[best_var]  

        elif self._choose_variables == "max_manhattan":

            chosen_vars = list(Y)

            # Select v variables with highest distances (most different)
            for _ in range(min(v, len(X_temp))):

                # Calculate Manhattan distance scores for each variable in X_temp
                var_scores = {}
                for var in X_temp:
                    # Get the dimensions of the current variable   
                    var_dims = get_var_dims(var)

                    # Calculate maximum Manhattan distance to any variable in Y
                    max_distance = 0
                    for y_var in chosen_vars:
                        # Get dimensions of the Y variable
                        y_dims = get_var_dims(y_var)

                        # Calculate Manhattan distance between var_dims and y_dims
                        distance = sum(abs(v1 - v2) for v1, v2 in zip(var_dims, y_dims))

                        # Keep track of maximum distance to any Y variable  
                        max_distance = max(max_distance, distance)
                    
                    # Store maximum Manhattan distance for this variable
                    var_scores[var] = max_distance

                best_var = max(var_scores.items(), key=lambda x: x[1])[0]
                chosen_vars.append(best_var)
                X_temp = set(X_temp) - {best_var}
                del var_scores[best_var]                
        elif self._choose_variables == "max_sum_manhattan":

            chosen_vars = list(Y)

            # Select v variables with highest distances (most different)
            for _ in range(min(v, len(X_temp))):

                # Calculate Manhattan distance scores for each variable in X_temp
                var_scores = {}
                for var in X_temp:
                    # Get the dimensions of the current variable   
                    var_dims = get_var_dims(var)

                    # Calculate sum of Manhattan distances to all variables in Y
                    total_distance = 0
                    for y_var in chosen_vars:
                        # Get dimensions of the Y variable
                        y_dims = get_var_dims(y_var)

                        # Calculate Manhattan distance between var_dims and y_dims
                        distance = sum(abs(v1 - v2) for v1, v2 in zip(var_dims, y_dims))

                        # Keep track of sum of distances to all Y variables  
                        total_distance += distance
                    
                    # Store sum of distances for this variable
                    var_scores[var] = total_distance

                best_var = max(var_scores.items(), key=lambda x: x[1])[0]
                chosen_vars.append(best_var)
                X_temp = set(X_temp) - {best_var}
                del var_scores[best_var]   

        elif self._choose_variables == "min_sum_manhattan":

            chosen_vars = list(Y)

            # Select v variables with highest distances (most different)
            for _ in range(min(v, len(X_temp))):

                # Calculate Manhattan distance scores for each variable in X_temp
                var_scores = {}
                for var in X_temp:
                    # Get the dimensions of the current variable   
                    var_dims = get_var_dims(var)

                    # Calculate sum of Manhattan distances to all variables in Y
                    total_distance = 0
                    for y_var in chosen_vars:
                        # Get dimensions of the Y variable
                        y_dims = get_var_dims(y_var)

                        # Calculate Manhattan distance between var_dims and y_dims
                        distance = sum(abs(v1 - v2) for v1, v2 in zip(var_dims, y_dims))

                        # Keep track of sum of distances to all Y variables  
                        total_distance += distance
                    
                    # Store sum of distances for this variable
                    var_scores[var] = total_distance

                best_var = min(var_scores.items(), key=lambda x: x[1])[0]
                chosen_vars.append(best_var)
                X_temp = set(X_temp) - {best_var}
                del var_scores[best_var]   

        elif self._choose_variables == "feature_based_par":
            
            # Get features from dataset and convert to numpy array for faster operations
            known_features = np.array(self.env.datasetX)
            chosen_vars = list(Y)

            # Handle empty features case
            if len(known_features) == 0 or len(Y) > len(X_temp):
                random.shuffle(X_temp)
                chosen_vars.extend([X_temp.pop() for _ in range(v)])
                return chosen_vars

            # Calculate feature ranges once, outside all loops
            feature_ranges = np.vstack((
                known_features.min(axis=0),
                known_features.max(axis=0)
            )).T  # Shape: (n_features, 2) where each row is [min, max]
            
            # Pre-calculate partitioning feature mask
            feature_names = list(self.env.feature_representation.get_feature_mapping().keys())
            is_partition = np.array([
                'Partition' in self.env.feature_representation.features[name] 
                for name in feature_names
            ])
            range_diff = feature_ranges[:, 1] - feature_ranges[:, 0]
            range_diff[range_diff == 0] = 1  # Avoid division by zero

            # Pre-calculate all possible constraints for each variable
            var_constraints = {
                var: get_con_subset(self.env.instance.bias, [var] + chosen_vars)
                for var in X_temp
            }

            best_var = None

            # Select v variables with highest scores
            for _ in range(min(v, len(X_temp))):
                var_scores = {}
                
                if best_var is not None:
                    var_constraints = {
                        var: var_constraints[var] + get_con_subset(self.env.instance.bias, [var, best_var])
                        for var in X_temp
                    }
                for var in X_temp:
                    # Create candidate constraints between this var and existing vars
                    candidate_constraints = var_constraints[var]
                    
                    if not var_constraints[var]:  # Handle empty constraints case
                        var_scores[var] = 0
                        continue
                        
                    feature_reprs = self.env.feature_representation.featurize_constraints(candidate_constraints)
                    feature_reprs = np.array(feature_reprs)
                    
                    # Vectorized similarity calculation
                    max_similarity = 0
                    for feature_repr in feature_reprs:
                        # Calculate all differences at once
                        diffs = np.abs(feature_repr - known_features)
                        
                        # Apply weights and normalization in a vectorized way
                        weighted_diffs = np.where(
                            is_partition,
                            10 * diffs,  # Partition features get 10x weight
                            diffs / range_diff  # Other features get normalized
                        )
                        # Sum along feature axis and convert to similarities
                        distances = weighted_diffs.sum(axis=1)
                        similarities = 1.0 / (1.0 + distances)
                        max_similarity = max(max_similarity, similarities.max())

                    # Store inverse similarity as score (so higher = more different)
                    var_scores[var] = 1.0 - max_similarity

                best_var = max(var_scores.items(), key=lambda x: x[1])[0]

                """print("\n is_partition: ", is_partition)
                # Print partitioning features for known constraints
                print("\nPartitioning features of known constraints:")
                for i, features in enumerate(known_features):
                    print(f"Constraint {i}: {features[is_partition]}")
                
                # Print partitioning features for best var's constraints
                print(f"\nPartitioning features of constraints involving {best_var}:")
                for i, constraint in enumerate(var_constraints[best_var]):
                    features = self.env.feature_representation.featurize_constraint(constraint)
                    print(f"Constraint {i}: {np.array(features)[is_partition]}")
                print("\nchosen_vars: ", chosen_vars)
                print("\nbest_var: ", best_var)
                print("\nvar_scores: ", var_scores)
                input("Press Enter to continue...")"""
                chosen_vars.append(best_var)
                X_temp = set(X_temp) - {best_var}
                del var_scores[best_var]

        elif self._choose_variables == "max_uncertainty":
            
            chosen_vars = list(Y)

            # Train classifier if we have learned constraints
            if len(self.env.instance.cl) > 0:
                self.env._train_classifier()
            self.env.predict_bias_proba_uncertanty()

            # Pre-calculate constraint entropies once
            constraint_entropy = {
                c: entropy([self.env.bias_proba_uncertanty[c], 1-self.env.bias_proba_uncertanty[c]], base=2)
                for c in self.env.instance.bias
            }

            # Cache related constraints for each variable
            var_constraints = {
                var: [c for c in self.env.instance.bias if var in set(get_scope(c))]
                for var in X_temp
            }

            # Select v variables with highest uncertainty scores
            for _ in range(min(v, len(X_temp))):
                # Calculate uncertainty scores for each variable in X_temp
                var_scores = {}
                for var in X_temp:
                    # Filter cached constraints to only those involving chosen variables
                    related_constraints = [
                        c for c in var_constraints[var]
                        if all(v in set(chosen_vars + [var]) for v in get_scope(c))
                    ]
                    
                    if related_constraints:
                        # Use pre-calculated entropies
                        var_scores[var] = max(constraint_entropy[c] for c in related_constraints)
                    else:
                        var_scores[var] = 0

                best_var = max(var_scores.items(), key=lambda x: x[1])[0]
                chosen_vars.append(best_var)
                X_temp = set(X_temp) - {best_var}
                del var_scores[best_var]
                del var_constraints[best_var]  # Clean up cache for selected variable.

        elif self._choose_variables == "sum_uncertainty":
            
            chosen_vars = list(Y)

            # Train classifier if we have learned constraints
            if len(self.env.instance.cl) > 0:
                self.env._train_classifier()
            self.env.predict_bias_proba_uncertanty()

            # Pre-calculate constraint entropies once
            constraint_entropy = {
                c: entropy([self.env.bias_proba_uncertanty[c], 1-self.env.bias_proba_uncertanty[c]], base=2)
                for c in self.env.instance.bias
            }

            # Cache related constraints for each variable
            var_constraints = {
                var: [c for c in self.env.instance.bias if var in set(get_scope(c))]
                for var in X_temp
            }

            # Select v variables with highest uncertainty scores
            for _ in range(min(v, len(X_temp))):
                # Calculate uncertainty scores for each variable in X_temp
                var_scores = {}
                for var in X_temp:
                    # Filter cached constraints to only those involving chosen variables
                    related_constraints = [
                        c for c in var_constraints[var]
                        if all(v in set(chosen_vars + [var]) for v in get_scope(c))
                    ]
                    
                    if related_constraints:
                        # Use pre-calculated entropies
                        var_scores[var] = sum(constraint_entropy[c] for c in related_constraints)
                    else:
                        var_scores[var] = 0

                best_var = max(var_scores.items(), key=lambda x: x[1])[0]
                chosen_vars.append(best_var)
                X_temp = set(X_temp) - {best_var}
                print("\nbest_var: ", best_var)
                print("\nvar_scores: ", var_scores)
                del var_scores[best_var]
                del var_constraints[best_var]  # Clean up cache for selected variable.
                print("\nvar_scores: ", var_scores)

        else: 
            raise ValueError(f"Invalid choose_variables strategy: {self._choose_variables}")
        
        return chosen_vars

