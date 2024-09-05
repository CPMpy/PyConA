import math
import cpmpy as cp

from ..utils import get_scope


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
    hashY = [hash(y) for y in Y]
    hashR = [hash(r) for r in R]

    x = cp.boolvar(shape=(len(Y),))

    model = cp.Model()

    constraints_Y1 = sum((1 - 10 * ((1 / P_c[i]) <= math.log2(len(Y)))) *
                         all(hash(scope_var) in hashR or x[hashY.index(hash(scope_var))]
                             for scope_var in get_scope(kappaB[i]))
                         for i in range(len(kappaB)))

    Y1_size = sum(x)

    model += Y1_size <= (len(Y) + 1) // 2
    model += Y1_size > 0

    model.maximize(constraints_Y1)

    model.solve()

    Y1 = [Y[i] for i in range(len(Y)) if x[i].value()]
    Y2 = list(set(Y) - set(Y1))

    return Y1, Y2
