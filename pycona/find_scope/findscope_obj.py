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
    :param P_c: Probabilities for the constraints in kappaB, as a dict keyed by constraint.
    :param kwargs: Additional keyword arguments. ``time_limit`` bounds the split optimization.
    :return: Two subsets of Y.
    """
    if len(kappaB) > 10000:
        return split_half(Y)

    time_limit = kwargs.get('time_limit', 1)

    hashY = [hash(y) for y in Y]
    hashR_set = set(hash(r) for r in R)
    hashY_to_idx = {h: i for i, h in enumerate(hashY)}

    x = cp.boolvar(shape=(len(Y),))

    model = cp.Model()

    constraints_Y1 = [
        cp.all(hash(scope_var) in hashR_set or x[hashY_to_idx[hash(scope_var)]]
               for scope_var in get_scope(k))
        for k in kappaB
    ]

    Y1_size = sum(x)

    model += Y1_size <= (len(Y) + 1) // 2
    model += Y1_size > 0

    # Weight is 1 - 10*p. Scale to an int: CPMpy cannot multiply a constraint by a float.
    weights = []
    for k in kappaB:
        p = P_c.get(k, 0.01) if isinstance(P_c, dict) else 0.01
        weights.append(int(round(100 * (1 - 10 * p))))
    objective = cp.sum([w * c for w, c in zip(weights, constraints_Y1)])

    model.maximize(objective)
    model.solve(time_limit=time_limit)

    Y1 = [Y[i] for i in range(len(Y)) if x[i].value()]
    Y2 = list(set(Y) - set(Y1))

    return Y1, Y2
