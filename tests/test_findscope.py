import pytest
from cpmpy.transformations.get_variables import get_variables
from cpmpy.transformations.normalize import toplevel_list

import pycona as ca
import cpmpy as cp

algorithms = [ca.FindScope(), ca.FindScope2()]


class TestFindScope:

    def test_findscope(self):
        a, b, c, d = cp.intvar(0, 9, shape=4)  # variables

        oracle_model = cp.Model()
        oracle_model += cp.AllDifferent([a, b, c, d]).decompose()
        oracle_model += 10 * c + d == 3 * (10 * a + b)
        oracle_model += 10 * d + a == 2 * (10 * b + c)
        constraints = toplevel_list(oracle_model.constraints)

        instance = ca.ProblemInstance(variables=cp.cpm_array([a, b, c, d]))
        ca_env = ca.ActiveCAEnv(find_scope=ca.FindScope())

        for c in range(len(constraints)):
            model = cp.Model(constraints[:c] + constraints[c+1:]) # all constraints except this
            model += ~constraints[c]
            model.solve()

            ca_env.init_state(oracle=ca.ConstraintOracle(oracle_model.constraints), instance=instance, verbose=1)
            Y = ca_env.run_find_scope([a, b, c, d])

            assert Y == get_variables(constraints[c])
