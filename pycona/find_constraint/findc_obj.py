import math

from .. import ActiveCAEnv
from ..ca_environment import ProbaActiveCAEnv
from ..utils import get_scope

def findc_obj_splithalf(sat, delta, **kwargs):
    """
    Objective function for FindC that aims to split the constraints in half.

    :param sat: The number of satisfied constraints.
    :param delta: The candidate constraints in the given scope.
    :return: The objective value.
    """
    return abs(sat - round(len(delta) / 2))

def findc_obj_proba(sat, delta, ca_env: ActiveCAEnv, **kwargs):
    """
    Probability-based objective function for FindC.

    :param sat: The number of satisfied constraints.
    :param delta: The candidate constraints in the given scope.
    :param ca_env: The constraint acquisition environment.
    :return: The objective value.
    :raises Exception: If the ca_environment is not an instance of ProbaActiveCAEnv.
    """
    if not isinstance(ca_env, ProbaActiveCAEnv):
        raise Exception('Probability based objective can only be used with ProbaActiveCAEnv')
    proba = {c: ca_env.bias_proba[c] for c in delta}

    Y = get_scope(delta[0])
    Y = list(dict.fromkeys(Y))  # Remove duplicates

    objective = sum(
        [~c * (1 - 2 * ((1 / proba[c]) <= math.log2(len(Y))) * sat) for
         c in delta])

    return objective
