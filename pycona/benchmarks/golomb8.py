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



def construct_golomb8():
    """
    :Description: The Golomb ruler problem is to place n marks on a ruler such that the distances between any two marks are all different.
    A Golomb ruler with 8 marks is sought in this instance.
    :return: a ProblemInstance object, along with a constraint-based oracle
    """
    # Parameters
    parameters = {"n_marks": 8}

    # Variables
    grid = cp.intvar(1, 35, shape=(1, 8), name="grid")

    C_T = []

    all_mark_pairs = []
    for a in range(8):
        for b in range(a + 1, 8):
            all_mark_pairs.append((a, b))

    for outer_idx in range(len(all_mark_pairs)):
        i, j = all_mark_pairs[outer_idx]  # Get the first pair of marks (i, j)

        for inner_idx in range(outer_idx + 1, len(all_mark_pairs)):
            x, y = all_mark_pairs[inner_idx]  # Get the second pair of marks (x, y)

            C_T += [cp.abs(grid[0, j] - grid[0, i]) != cp.abs(grid[0, y] - grid[0, x])]

    for i in range(8):
        for j in range(i + 1, 8):
            C_T += [grid[0, i] < grid[0, j]]

    # Create the language:
    AV = absvar(4)  # create abstract vars - as many as maximum arity

    # create abstract relations using the abstract vars
    lang = [AV[0] == AV[1], AV[0] != AV[1], AV[0] < AV[1], AV[0] > AV[1], AV[0] >= AV[1], AV[0] <= AV[1], cp.abs(AV[0] - AV[1]) != cp.abs(AV[2] - AV[3])]

    instance = GolombInstance(variables=grid, params=parameters, language=lang, name="golomb8")

    oracle = ConstraintOracle(list(set(toplevel_list(C_T))))

    return instance, oracle