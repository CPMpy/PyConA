import copy
from itertools import combinations

import cpmpy as cp
from cpmpy.expressions.core import Expression, Comparison, Operator
from cpmpy.expressions.variables import NDVarArray, _NumVarImpl, NegBoolView
from cpmpy.transformations.get_variables import get_variables
from sklearn.utils import class_weight
import numpy as np
import re
from cpmpy.expressions.utils import all_pairs, is_any_list


class Objectives:
    """
    A class to manage different objectives for query generation, find scope, and find constraint.
    """

    @classmethod
    def qgen_objectives(cls):
        """
        Get the list of query generation objectives.

        :return: List of query generation objectives.
        """
        from .query_generation.qgen_obj import obj_max_viol, obj_min_viol, obj_proba
        return [obj_max_viol, obj_min_viol, obj_proba]

    @classmethod
    def findscope_objectives(cls):
        """
        Get the list of find scope objectives.

        :return: List of find scope objectives.
        """
        from .find_scope.findscope_obj import split_half, split_proba
        return [split_half, split_proba]

    @classmethod
    def findc_objectives(cls):
        """
        Get the list of find constraint objectives.

        :return: List of find constraint objectives.
        """
        from .find_constraint.findc_obj import findc_obj_splithalf, findc_obj_proba
        return [findc_obj_splithalf, findc_obj_proba]


def check_value(c):
    """
    Check the value of a constraint.

    :param c: The constraint to check.
    :return: Boolean value of the constraint.
    """
    return bool(c.value())


def get_con_subset(B, Y):
    """
    Get the subset of constraints whose scope is a subset of Y.

    :param B: List of constraints.
    :param Y: Set of variables.
    :return: List of constraints whose scope is a subset of Y.
    """
    Y = frozenset(Y)
    return [c for c in B if frozenset(get_variables(c)).issubset(Y)]


def get_kappa(B, Y):
    """
    Get the subset of constraints whose scope is a subset of Y and are not satisfied.

    :param B: List of constraints.
    :param Y: Set of variables.
    :return: List of constraints whose scope is a subset of Y and are not satisfied.
    """
    Y = frozenset(Y)
    return [c for c in B if frozenset(get_variables(c)).issubset(Y) and check_value(c) is False]


def get_lambda(B, Y):
    """
    Get the subset of constraints whose scope is a subset of Y and are satisfied.

    :param B: List of constraints.
    :param Y: Set of variables.
    :return: List of constraints whose scope is a subset of Y and are satisfied.
    """
    Y = frozenset(Y)
    return [c for c in B if frozenset(get_variables(c)).issubset(Y) and check_value(c) is True]


def gen_pairwise(v1, v2):
    """
    Generate pairwise constraints between two variables.

    :param v1: First variable.
    :param v2: Second variable.
    :return: List of pairwise constraints.
    """
    return [v1 == v2, v1 != v2, v1 < v2, v1 > v2]


def gen_pairwise_ineq(v1, v2):
    """
    Generate pairwise inequality constraints between two variables.

    :param v1: First variable.
    :param v2: Second variable.
    :return: List of pairwise inequality constraints.
    """
    return [v1 != v2]


def alldiff_binary(grid):
    """
    Generate all different binary constraints for a grid.

    :param grid: The grid of variables.
    :yield: All different binary constraints.
    """
    for v1, v2 in all_pairs(grid):
        for c in gen_pairwise_ineq(v1, v2):
            yield c


def gen_scoped_cons(grid):
    """
    Generate scoped constraints for a grid.

    :param grid: The grid of variables.
    :yield: Scoped constraints.
    """
    # rows
    for row in grid:
        for v1, v2 in all_pairs(row):
            for c in gen_pairwise_ineq(v1, v2):
                yield c
    # columns
    for col in grid.T:
        for v1, v2 in all_pairs(col):
            for c in gen_pairwise_ineq(v1, v2):
                yield c
    # subsquares
    for i1 in range(0, 4, 2):
        for i2 in range(i1, i1 + 2):
            for j1 in range(0, 4, 2):
                for j2 in range(j1, j1 + 2):
                    if (i1 != i2 or j1 != j2):
                        for c in gen_pairwise_ineq(grid[i1, j1], grid[i2, j2]):
                            yield c


def gen_all_cons(grid):
    """
    Generate all pairwise constraints for a grid.

    :param grid: The grid of variables.
    :yield: All pairwise constraints.
    """
    for v1, v2 in all_pairs(grid.flat):
        for c in gen_pairwise(v1, v2):
            yield c


def get_scopes_vars(C):
    """
    Get the set of variables involved in the scopes of constraints.

    :param C: List of constraints.
    :return: Set of variables involved in the scopes of constraints.
    """
    return set([x for scope in [get_variables(c) for c in C] for x in scope])


def get_scopes(C):
    """
    Get the list of unique scopes of constraints.

    :param C: List of constraints.
    :return: List of unique scopes of constraints.
    """
    return list(set([tuple(get_variables(c)) for c in C]))


def get_scope(constraint):
    """
    Get the scope (variables) of a constraint.

    :param constraint: The constraint to get the scope of.
    :return: List of variables in the scope of the constraint.
    """
    return get_variables(constraint)
        
        
def compare_scopes(scope1, scope2):

    scope1 = set(scope1)
    scope2 = set(scope2)

    if len(scope1) != len(scope2):
        return False

    return all(v in scope2 for v in scope1)


def get_constant(constraint):
    """
    Get the constants involved in a constraint.

    :param constraint: The constraint to get the constants of.
    :return: List of constants involved in the constraint.
    """

    if isinstance(constraint, _NumVarImpl):
        return []
    elif isinstance(constraint, Expression) or is_any_list(constraint):
        constants = []
        for argument in (constraint.args if isinstance(constraint, Expression) else constraint):
            if not isinstance(argument, _NumVarImpl):
                constants.extend(get_constant(argument))
        return constants
    else:
        return [constraint]


def get_arity(constraint):
    """
    Get the arity (number of variables) of a constraint.

    :param constraint: The constraint to get the arity of.
    :return: The arity of the constraint.
    """
    return len(get_scope(constraint))


def get_min_arity(C):
    """
    Get the minimum arity of a list of constraints.

    :param C: List of constraints.
    :return: The minimum arity of the constraints.
    """
    if len(C) > 0:
        return min([get_arity(c) for c in C])
    return 0


def get_max_arity(C):
    """
    Get the maximum arity of a list of constraints.

    :param C: List of constraints.
    :return: The maximum arity of the constraints.
    """
    if len(C) > 0:
        return max([get_arity(c) for c in C])
    return 0


def get_relation(c, gamma):
    """
    Get the relation index of a constraint in a given language.

    :param c: The constraint.
    :param gamma: The language (list of relations).
    :return: The index of the relation in the language, or -1 if not found.
    """
    scope = get_scope(c)
    for i in range(len(gamma)):
        relation = gamma[i]
        abs_vars = get_scope(relation)
        if len(abs_vars) != len(scope):
            continue

        replace_dict = dict()
        for idx, var in enumerate(scope):
            replace_dict[abs_vars[idx]] = var

        constraint = replace_variables(relation, replace_dict)

        if hash(constraint) == hash(c):
            return i

    return -1


def replace_variables(constraint, var_mapping):
    """
    Replace the variables in a constraint using a dictionary mapping previous variables to new ones.

    :param constraint: The constraint to replace the variables in.
    :param var_mapping: Dictionary mapping previous variables to new ones.
    :return: The new expression with the variables replaced.
    """
    if isinstance(constraint, _NumVarImpl):
        return var_mapping.get(constraint, constraint)
    elif isinstance(constraint, Expression):
        # Create a shallow copy of the original expression
        new_constraint = copy.copy(constraint)
        new_args = []
        for argument in constraint.args:
            if isinstance(argument, _NumVarImpl):
                new_args.append(var_mapping.get(argument, argument))
            else:
                new_args.append(replace_variables(argument, var_mapping))
        # Replace the arguments in the copied expression
        new_constraint.update_args(new_args)
        return new_constraint
    else:
        return constraint


def get_var_name(var):
    """
    Get the name of a variable without its indices.

    :param var: The variable.
    :return: The name of the variable without its indices.
    """
    name = re.findall(r"\[\d+[,\d+]*\]", var.name)
    if name:  # Check if we found any indices
        return var.name.replace(name[0], '')
    return var.name  # Return original name if no indices found


def get_var_ndims(var):
    """
    Get the number of dimensions of a variable.

    :param var: The variable.
    :return: The number of dimensions of the variable.
    """
    dims = re.findall(r"\[\d+[,\d+]*\]", var.name)
    dims_str = "".join(dims)
    ndims = len(re.split(",", dims_str))
    return ndims


def get_var_dims(var):
    """
    Get the dimensions of a variable.

    :param var: The variable.
    :return: The dimensions of the variable. Returns empty list if variable has no indices.
    """
    dims = re.findall(r"\[\d+[,\d+]*\]", var.name)
    if not dims:  # If no indices found
        return []
    dims_str = "".join(dims)
    dims = re.split(r"[\[\]]", dims_str)[1]
    dims = [int(dim) for dim in re.split(",", dims)]
    return dims


def get_divisors(n):
    """
    Get the divisors of a number.

    :param n: The number.
    :return: List of divisors of the number.
    """
    divisors = list()
    for i in range(2, int(n / 2) + 1):
        if n % i == 0:
            divisors.append(i)
    return divisors


def average_difference(values):
    """
    Calculate the average difference between consecutive values in a list.

    :param values: List of values.
    :return: The average difference between consecutive values.
    """
    if len(values) < 2:
        return 0
    differences = [values[i + 1] - values[i] for i in range(len(values) - 1)]
    return sum(differences) / len(differences)


def compute_sample_weights(Y):
    """
    Compute sample weights for a list of labels.

    :param Y: List of labels.
    :return: List of sample weights.
    """
    c_w = class_weight.compute_class_weight('balanced', classes=np.unique(Y), y=Y)
    sw = []

    for i in range(len(Y)):
        if Y[i] == False:
            sw.append(c_w[0])
        else:
            sw.append(c_w[1])

    return sw


def get_variables_from_constraints(constraints):
    """
    Get the list of variables involved in a list of constraints.

    :param constraints: List of constraints.
    :return: List of variables involved in the constraints.
    """
    # Create set to hold unique variables
    variable_set = set()
    for constraint in constraints:
        variable_set.update(get_variables(constraint))

    def extract_nums(s):
        dims = re.findall(r"\[\d+[,\d+]*\]", s.name)
        if not dims:
            return [0]  # Default value for variables without indices
        dims_str = "".join(dims)
        dims = re.split(r"[\[\]]", dims_str)[1]
        return [int(dim) for dim in re.split(",", dims)]

    variable_list = sorted(variable_set, key=extract_nums)
    return variable_list


def combine_sets_distinct(set1, set2):
    """
    Combine two sets into a set of distinct pairs.

    :param set1: First set.
    :param set2: Second set.
    :return: Set of distinct pairs.
    """
    result = set()
    examined = set()
    for a in set1:
        examined.add(a)
        for b in set2:
            if a is not b and b not in examined:
                # Add tuple of sorted combinations to remove duplicates in different order
                result.add(tuple(sorted((a, b))))
    return result


def unravel(lst, newlist):
    """
    Recursively unravel nested lists, tuples, or arrays into a flat list.

    :param lst: The nested list, tuple, or array to unravel.
    :param newlist: The flat list to append the elements to.
    """
    for e in lst:
        if isinstance(e, Expression):
            if isinstance(e, NDVarArray):
                unravel(e.flat, newlist)
            else:
                newlist.append(e)
        elif isinstance(e, (list, tuple, np.flatiter, np.ndarray)):
            unravel(e, newlist)


def get_combinations(lst, n):
    """
    Get all combinations of a list of a given length.

    :param lst: The list.
    :param n: The length of combinations.
    :return: List of combinations.
    """
    if is_any_list(lst):
        newlist = []
        unravel(lst, newlist)
        lst = newlist
    return list(combinations(lst, n))


def restore_scope_values(scope, scope_values):
    """
    Restore the original values of variables in a scope.

    :param scope: The scope of variables.
    :param scope_values: The original values of the variables.
    :return: None
    """
    tmp = cp.Model()
    i = 0
    for x in scope:
        tmp += x == scope_values[i]
        i = i + 1

    tmp.solve()

    return
