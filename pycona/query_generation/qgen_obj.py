import math

from .. import ActiveCAEnv
from ..ca_environment import ProbaActiveCAEnv
from ..utils import get_variables_from_constraints


def obj_max_viol(B, **kwargs):
    """
    Objective function to maximize the number of violated constraints.

    :param B: A list of constraints.
    :return: The sum of violated constraints.
    """
    return sum([~c for c in B])


def obj_min_viol(B, **kwargs):
    """
    Objective function to minimize the number of violated constraints.

    :param B: A list of constraints.
    :return: The sum of satisfied constraints.
    """
    return sum([c for c in B])


def obj_proba(B, ca_env: ActiveCAEnv, **kwargs):
    """
    Probability-based objective function.

    :param B: A list of constraints.
    :param ca_env: An instance of ActiveCAEnv.
    :return: The probability-based objective function.
    :raises Exception: If ca_environment is not an instance of ProbaActiveCAEnv.
    """
    if not isinstance(ca_env, ProbaActiveCAEnv):
        raise Exception('Probability based objective can only be used with ProbaActiveCAEnv')

    proba = {c: ca_env.bias_proba[c] for c in B}
    Y = get_variables_from_constraints(B)

    O_c = [((1 / proba[c]) <= math.log2(len(Y))) for c in B]
    objective = sum(
        [~c * (1 - len(ca_env.instance.language) * o_c) for
         c, o_c in zip(B, O_c)])

    return objective
