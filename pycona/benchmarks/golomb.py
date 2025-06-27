import cpmpy as cp
from cpmpy.transformations.normalize import toplevel_list
from ..answering_queries.constraint_oracle import ConstraintOracle
from ..problem_instance import ProblemInstance, absvar
from itertools import combinations
from ..utils import get_scope, replace_variables, combine_sets_distinct

class GolombInstance(ProblemInstance):

    def construct_bias(self, X=None):
        """
        Construct the bias (candidate constraints) for the golomb instance. 
        We need a different bias construction for the golomb instance because 
        it needs to include all permutations of scopes for the quaternary relations.
        """
        if X is None:
            X = self.X

        all_cons = []

        for relation in self.language:

            abs_vars = get_scope(relation)

            combs = list(combinations(X, 2))

            if len(abs_vars) == 2:
                for comb in combs:
                    replace_dict = dict()
                    for i, v in enumerate(comb):
                        replace_dict[abs_vars[i]] = v
                    constraint = replace_variables(relation, replace_dict)
                    all_cons.append(constraint)
            elif len(abs_vars) == 4:
                result_combinations = combine_sets_distinct(combs, combs)
                for ((v1, v2), (v3, v4)) in result_combinations:
                    replace_dict = dict()
                    replace_dict[abs_vars[0]] = v1
                    replace_dict[abs_vars[1]] = v2
                    replace_dict[abs_vars[2]] = v3
                    replace_dict[abs_vars[3]] = v4
                    constraint = replace_variables(relation, replace_dict)
                    all_cons.append(constraint)

        self.bias = list(set(all_cons) - set(self.cl) - set(self.excluded_cons))

    def construct_bias_for_vars(self, v1, X=None):
        """
        Construct the bias (candidate constraints) for specific variables in the golomb instance.
        Overrides the parent class method to handle the special case of quaternary relations in Golomb.
        
        Args:
            v1: The variable(s) for which to construct the bias. Can be a single variable or list of variables.
            X: The set of variables to consider, default is None (uses self.X).
        """
        if not isinstance(v1, list):
            v1 = [v1]

        if X is None:
            X = self.X
            
        # Sort X based on variable names for consistency
        X = sorted(X, key=lambda var: var.name)

        all_cons = []

        for relation in self.language:
            abs_vars = get_scope(relation)

            combs = list(combinations(X, 2))

            if len(abs_vars) == 2:
                for comb in combs:
                    replace_dict = dict()
                    for i, v in enumerate(comb):
                        replace_dict[abs_vars[i]] = v
                    constraint = replace_variables(relation, replace_dict)
                    all_cons.append(constraint)
            elif len(abs_vars) == 4:
                result_combinations = combine_sets_distinct(combs, combs)
                for ((v1_, v2), (v3, v4)) in result_combinations:
                    replace_dict = dict()
                    replace_dict[abs_vars[0]] = v1_
                    replace_dict[abs_vars[1]] = v2
                    replace_dict[abs_vars[2]] = v3
                    replace_dict[abs_vars[3]] = v4
                    constraint = replace_variables(relation, replace_dict)
                    all_cons.append(constraint)

        # Filter constraints to only include those containing at least one of the specified variables
        filtered_cons = [c for c in all_cons if any(v in set(get_scope(c)) for v in v1)]
        self.bias = list(set(filtered_cons) - set(self.cl) - set(self.excluded_cons))


def construct_golomb(n_marks=8):
    """
    :Description: The Golomb ruler problem is to place n marks on a ruler such that the distances between any two marks are all different.
    A Golomb ruler with 8 marks is sought in this instance.
    :return: a ProblemInstance object, along with a constraint-based oracle
    """
    # Parameters
    parameters = {"n_marks": n_marks}

    # Variables
    grid = cp.intvar(1, n_marks*4, shape=(1, n_marks), name="grid")

    C_T = []

    all_mark_pairs = []
    for a in range(n_marks):
        for b in range(a + 1, n_marks):
            all_mark_pairs.append((a, b))

    for outer_idx in range(len(all_mark_pairs)):
        i, j = all_mark_pairs[outer_idx]  # Get the first pair of marks (i, j)

        for inner_idx in range(outer_idx + 1, len(all_mark_pairs)):
            x, y = all_mark_pairs[inner_idx]  # Get the second pair of marks (x, y)

            C_T += [cp.abs(grid[0, j] - grid[0, i]) != cp.abs(grid[0, y] - grid[0, x])]

    for i in range(n_marks):
        for j in range(i + 1, n_marks):
            C_T += [grid[0, i] < grid[0, j]]

    # Create the language:
    AV = absvar(4)  # create abstract vars - as many as maximum arity

    # create abstract relations using the abstract vars
    lang = [AV[0] == AV[1], AV[0] != AV[1], AV[0] < AV[1], AV[0] > AV[1], AV[0] >= AV[1], AV[0] <= AV[1], cp.abs(AV[0] - AV[1]) != cp.abs(AV[2] - AV[3])]

    instance = GolombInstance(variables=grid, params=parameters, language=lang, name="golomb")

    oracle = ConstraintOracle(list(set(toplevel_list(C_T))))

    print("Target constraints: ", len(oracle.constraints))
    return instance, oracle