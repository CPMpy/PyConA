import pytest
from pycona.ca_environment import ProbaActiveCAEnv
from pycona.benchmarks import construct_sudoku


class TestProbaActiveCAEnv:
    @pytest.mark.fast
    def test_add_to_cl_skips_learned_constraints(self):
        """Test that constraints already in CL are not added again to CL or to the dataset"""
        instance, oracle = construct_sudoku(2, 2, 4)
        instance.construct_bias()
        ca_env = ProbaActiveCAEnv()
        ca_env.init_state(instance, oracle, verbose=0)

        c1, c2 = ca_env.instance.bias[:2]
        ca_env.add_to_cl([c1, c1])
        ca_env.add_to_cl([c1, c2])

        assert ca_env.instance.cl.count(c1) == 1
        assert ca_env.instance.cl.count(c2) == 1
        assert len(ca_env.datasetX) == len(ca_env.datasetY) == 2
