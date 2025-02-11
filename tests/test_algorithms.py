import unittest

from itertools import product
import pytest
import pycona as ca

from pycona.benchmarks import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

problem_generators = [construct_murder_problem(), construct_examtt_simple(6, 3, 2, 10), construct_nurse_rostering()]

benchmarks = []
instance, oracle = construct_murder_problem()
benchmarks.append({"instance": instance, "oracle": oracle})
instance, oracle = construct_examtt_simple(6, 3, 2, 10)
benchmarks.append({"instance": instance, "oracle": oracle})
instance, oracle = construct_nurse_rostering()
benchmarks.append({"instance": instance, "oracle": oracle})

classifiers = [DecisionTreeClassifier(), RandomForestClassifier()]
algorithms = [ca.QuAcq(), ca.MQuAcq(), ca.MQuAcq2(), ca.GQuAcq(), ca.PQuAcq()]

def _generate_benchmarks():
    for generator in problem_generators:
        yield tuple(generator)


def _generate_base_inputs():
    combs = product(_generate_benchmarks(), algorithms)
    for comb in combs:
        yield comb


def _generate_proba_inputs():
    combs = product(_generate_benchmarks(), algorithms, classifiers)
    for comb in combs:
        yield comb


class TestAlgorithms:

    @pytest.mark.parametrize(("bench", "algorithm"), _generate_base_inputs(), ids=str)
    def test_base_algorithms(self, bench, algorithm):
        (instance, oracle) = bench
        ca_system = algorithm
        learned_instance = ca_system.learn(instance=instance, oracle=oracle)
        assert len(learned_instance.cl) > 0
        assert learned_instance.get_cpmpy_model().solve()

    @pytest.mark.parametrize(("bench", "inner_alg"), _generate_base_inputs(), ids=str)
    def test_growacq(self, bench, inner_alg):
        env = ca.ActiveCAEnv()
        (instance, oracle) = bench
        ca_system = ca.GrowAcq(env, inner_algorithm=inner_alg)
        learned_instance = ca_system.learn(instance=instance, oracle=oracle)
        assert len(learned_instance.cl) > 0
        assert learned_instance.get_cpmpy_model().solve()

    @pytest.mark.parametrize(("bench", "algorithm", "classifier"), _generate_proba_inputs(), ids=str)
    def test_proba(self, bench, algorithm, classifier):
        env = ca.ProbaActiveCAEnv(classifier=classifier)
        (instance, oracle) = bench
        ca_system = algorithm
        ca_system.env = env
        learned_instance = ca_system.learn(instance=instance, oracle=oracle)
        assert len(learned_instance.cl) > 0
        assert learned_instance.get_cpmpy_model().solve()

    @pytest.mark.parametrize(("bench", "inner_alg", "classifier"), _generate_proba_inputs(), ids=str)
    def test_proba_growacq(self, bench, inner_alg, classifier):
        env = ca.ProbaActiveCAEnv(classifier=classifier)
        (instance, oracle) = bench
        ca_system = ca.GrowAcq(env, inner_algorithm=inner_alg)
        learned_instance = ca_system.learn(instance=instance, oracle=oracle)
        assert len(learned_instance.cl) > 0
        assert learned_instance.get_cpmpy_model().solve()

    @pytest.mark.parametrize(("bench", "algorithm"), _generate_base_inputs(), ids=str)
    def test_base_algorithms_with_initial_cl(self, bench, algorithm):
        (instance, oracle) = bench
        # Create a copy of the instance to avoid modifying the original
        instance = instance.copy()
        
        # Get some constraints from the oracle's constraint set
        initial_constraints = oracle.constraints[:len(oracle.constraints)//2]  # Take half of the constraints
        instance.cl.extend(initial_constraints)
        initial_cl_size = len(instance.cl)
        
        ca_system = algorithm
        learned_instance = ca_system.learn(instance=instance, oracle=oracle)
        assert len(learned_instance.cl) == initial_cl_size*2
        assert learned_instance.get_cpmpy_model().solve()

    @pytest.mark.parametrize(("bench", "algorithm", "classifier"), _generate_proba_inputs(), ids=str)
    def test_proba_with_initial_cl(self, bench, algorithm, classifier):
        env = ca.ProbaActiveCAEnv(classifier=classifier)
        (instance, oracle) = bench
        # Create a copy of the instance to avoid modifying the original
        instance = instance.copy()
        
        # Get some constraints from the oracle's constraint set
        initial_constraints = oracle.constraints[:len(oracle.constraints)//2]  # Take half of the constraints
        instance.cl.extend(initial_constraints)
        initial_cl_size = len(instance.cl)
        
        ca_system = algorithm
        ca_system.env = env
        learned_instance = ca_system.learn(instance=instance, oracle=oracle)
        assert len(learned_instance.cl) == initial_cl_size*2
        assert learned_instance.get_cpmpy_model().solve()

    @pytest.mark.parametrize(("bench", "algorithm"), _generate_base_inputs(), ids=str)
    def test_base_algorithms_with_bias(self, bench, algorithm):
        (instance, oracle) = bench
        # Create a copy of the instance to avoid modifying the original
        instance = instance.copy()
        
        # Generate bias constraints for the instance
        instance.construct_bias()
        # Separate constraints into those from oracle and others
        oracle_constraints = set(oracle.constraints)
        other_constraints = [c for c in instance.bias if c not in oracle_constraints]
        # Keep all oracle constraints and half of the other constraints
        instance.bias = list(oracle_constraints) + other_constraints[:len(other_constraints)//2]
        
        ca_system = algorithm
        learned_instance = ca_system.learn(instance=instance, oracle=oracle)
        assert len(learned_instance.cl) > 0
        assert learned_instance.get_cpmpy_model().solve()

    @pytest.mark.parametrize(("bench", "algorithm", "classifier"), _generate_proba_inputs(), ids=str)
    def test_proba_with_bias(self, bench, algorithm, classifier):
        env = ca.ProbaActiveCAEnv(classifier=classifier)
        (instance, oracle) = bench
        # Create a copy of the instance to avoid modifying the original
        instance = instance.copy()
        
        # Generate bias constraints for the instance
        instance.construct_bias()
        # Separate constraints into those from oracle and others
        oracle_constraints = set(oracle.constraints)
        other_constraints = [c for c in instance.bias if c not in oracle_constraints]
        # Keep all oracle constraints and half of the other constraints
        instance.bias = list(oracle_constraints) + other_constraints[:len(other_constraints)//2]
        
        ca_system = algorithm
        ca_system.env = env
        learned_instance = ca_system.learn(instance=instance, oracle=oracle)
        assert len(learned_instance.cl) > 0
        assert learned_instance.get_cpmpy_model().solve()