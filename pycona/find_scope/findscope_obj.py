import math
import cpmpy as cp

from ..utils import get_scope, restore_scope_values


def split_half(Y, **kwargs):
    """
    Split the set Y into two halves.

    :param Y: A list of variables.
    :param kwargs: Additional keyword arguments.
    :return: Two halves of the list Y.
    """
    s = len(Y) // 2
    Y1, Y2 = Y[:s], Y[s:]
    return Y1, Y2


def split_proba(Y, R, kappaB, P_c, **kwargs):
    """
    Split the set Y based on probabilities for constraints.

    :param Y: A list of variables.
    :param R: A set of variables.
    :param kappaB: A list of constraints.
    :param P_c: A list of probabilities.
    :param kwargs: Additional keyword arguments.
    :return: Two subsets of Y.
    """
    if len(kappaB) > 10000:
        return split_half(Y)
    
    hashY = [hash(y) for y in Y]
    hashR = [hash(r) for r in R]

    x = cp.boolvar(shape=(len(Y),))

    model = cp.Model()

    Y1_size = sum(x)

    model += Y1_size <= (len(Y) + 1) // 2
    model += Y1_size > 0

    s = cp.SolverLookup.get("ortools", model)

    if not s.solve():
        raise Exception("No solution found in split_proba of findscope, please report an issue")

    # Next solve will change the values of the variables in x
    # so we need to return them to the original ones to continue if we don't find a solution next
    values = [x.value() for x in x]

    # So a solution was found, try to find a better one now
    s.solution_hint(x, values)

    constraints_Y1 = [cp.all(hash(scope_var) in set(hashR) or x[hashY.index(hash(scope_var))]
                             for scope_var in get_scope(kappaB[i]))
                         for i in range(len(kappaB))]
    
    objective = sum((1 - 10 * P_c[kappaB[i]]) *
                    constraints_Y1[i]
                    for i in range(len(kappaB)))
    
    s.maximize(objective)
    
    flag = s.solve(time_limit=1)

    if not flag:
        restore_scope_values(x, values)
    
    Y1 = [Y[i] for i in range(len(Y)) if x[i].value()]
    Y2 = list(set(Y) - set(Y1))
    
    return Y1, Y2
