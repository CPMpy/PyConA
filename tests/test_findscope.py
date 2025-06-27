import pytest
from cpmpy.transformations.get_variables import get_variables
from cpmpy.transformations.normalize import toplevel_list

import pycona as ca
import cpmpy as cp

algorithms = [ca.FindScope(), ca.FindScope2()]
fast_algorithms = [ca.FindScope2()]  # Use only FindScope for fast tests


class TestFindScope:

    @pytest.mark.parametrize(
        "algorithm",
        [
            *[pytest.param(alg, marks=pytest.mark.fast) for alg in fast_algorithms],
            *[pytest.param(alg) for alg in algorithms if alg not in fast_algorithms]
        ]
    )
    def test_findscope(self, algorithm):
        a, b, c, d = cp.intvar(0, 9, shape=4)  # variables
        vars_array = cp.cpm_array([a, b, c, d])

        oracle_model = cp.Model()
        oracle_model += cp.AllDifferent(vars_array).decompose()
        oracle_model += 10 * c + d == 3 * (10 * a + b)
        oracle_model += 10 * d + a == 2 * (10 * b + c)
        constraints = toplevel_list(oracle_model.constraints)

        instance = ca.ProblemInstance(variables=cp.cpm_array(vars_array))
        ca_env = ca.ActiveCAEnv(find_scope=algorithm)

        for con in range(len(constraints)):
            model = cp.Model(constraints[:con] + constraints[con + 1:])  # all constraints except this
            model += ~constraints[con]

            if not model.solve():
                continue

            ca_env.init_state(oracle=ca.ConstraintOracle(oracle_model.constraints), instance=instance, verbose=1)
            Y = ca_env.run_find_scope(vars_array)

            assert ca.utils.compare_scopes(Y, get_variables(constraints[con])), f"{Y}, {get_variables(constraints[con])}"
