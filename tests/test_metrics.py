import pytest
from pycona.ca_environment import ActiveCAEnv
from pycona.benchmarks import construct_sudoku
from pycona.utils import get_relation


class TestMetrics:
    @pytest.mark.fast
    def test_generalization_queries_log(self):
        """Test that generalization queries are logged in the metrics along with their answers"""
        instance, oracle = construct_sudoku(2, 2, 4)
        instance.construct_bias()
        ca_env = ActiveCAEnv()
        ca_env.init_state(instance, oracle, verbose=0)

        c_true = oracle.constraints[0]
        c_false = next(c for c in ca_env.instance.bias if c not in set(oracle.constraints))
        for c in [c_true, c_false]:
            ca_env.ask_generalization_query(ca_env.instance.language[get_relation(c, ca_env.instance.language)], [c])

        metrics = ca_env.metrics
        assert metrics.generalization_queries == [([c_true], True), ([c_false], False)]
        assert metrics.generalization_queries_count == 2
        assert metrics.positive_generalization_queries == 1
