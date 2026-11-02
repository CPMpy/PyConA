import pytest
from sklearn.tree import DecisionTreeClassifier
from pycona.ca_environment import ProbaActiveCAEnv
from pycona.predictor import CountsPredictor
from pycona.benchmarks import construct_sudoku

classifiers = [DecisionTreeClassifier, CountsPredictor]


class TestPredictBiasProba:
    @pytest.mark.fast
    @pytest.mark.parametrize("classifier", classifiers)
    def test_predict_bias_proba(self, classifier):
        """Test that the probabilities of the candidates are the ones predicted for each constraint"""
        instance, oracle = construct_sudoku(2, 2, 4)
        instance.construct_bias()
        ca_env = ProbaActiveCAEnv(classifier=classifier())
        ca_env.init_state(instance, oracle, verbose=0)

        bias = ca_env.instance.bias
        ca_env.add_to_cl(bias[:5])
        ca_env.remove_from_bias(bias[5:10])
        ca_env._train_classifier()
        ca_env._predict_bias_proba()

        for c in ca_env.instance.bias:
            features = ca_env.feature_representation.featurize_constraint(c)
            assert ca_env.bias_proba[c] == pytest.approx(ca_env.classifier.predict_proba([features])[0][1] + 0.01)

    @pytest.mark.fast
    def test_predict_bias_proba_empty_bias(self):
        """Test that predicting with an empty bias does not fail"""
        instance, oracle = construct_sudoku(2, 2, 4)
        instance.construct_bias()
        ca_env = ProbaActiveCAEnv()
        ca_env.init_state(instance, oracle, verbose=0)

        bias = ca_env.instance.bias
        ca_env.add_to_cl(bias[:5])
        ca_env.remove_from_bias(bias[5:])
        ca_env._train_classifier()
        ca_env._predict_bias_proba()

        assert ca_env.bias_proba == {}
