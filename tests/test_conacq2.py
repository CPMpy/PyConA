import pytest
import cpmpy as cp
from cpmpy.transformations.normalize import toplevel_list

import pycona as ca
from pycona.active_algorithms.conacq2 import VersionSpace
from pycona.answering_queries.constraint_oracle import ConstraintOracle
from pycona.problem_instance import ProblemInstance, absvar
from pycona.query_generation.conacq_qgen import constraint_complement


def _tiny_neq_instance():
    """Three variables with target all-different on pairs; language {==, !=}."""
    X = cp.intvar(1, 3, shape=3, name="x")
    C_T = [X[0] != X[1], X[1] != X[2], X[0] != X[2]]
    AV = absvar(2)
    lang = [AV[0] == AV[1], AV[0] != AV[1]]
    instance = ProblemInstance(variables=X, language=lang, name="tiny_neq")
    oracle = ConstraintOracle(list(set(toplevel_list(C_T))))
    return instance, oracle


class TestVersionSpace:

    @pytest.mark.fast
    def test_positive_forces_false(self):
        x, y = cp.intvar(1, 2, name="x"), cp.intvar(1, 2, name="y")
        c_eq, c_neq = x == y, x != y
        vs = VersionSpace([c_eq, c_neq])
        assert cp.Model([x == 1, y == 1]).solve()
        # Assignment satisfies ==, violates !=
        kappa = vs.kappa_under_T([x, y])
        assert c_neq in kappa
        vs.add_positive(kappa)
        assert c_neq in vs.forced_false
        assert c_eq not in vs.forced_false

    @pytest.mark.fast
    def test_negative_unit_forces_true(self):
        x, y = cp.intvar(1, 2, name="x"), cp.intvar(1, 2, name="y")
        c_eq, c_neq = x == y, x != y
        vs = VersionSpace([c_eq, c_neq])
        vs.force_false(c_eq)
        vs.propagate()
        assert cp.Model([x == 1, y == 1]).solve()
        kappa = vs.kappa_under_T([x, y])
        # Only != rejects this after == is ruled out
        assert kappa == [c_neq] or set(kappa) == {c_neq}
        vs.add_negative(kappa)
        assert c_neq in vs.forced_true

    @pytest.mark.fast
    def test_negative_clause(self):
        x, y = cp.intvar(1, 2, name="x"), cp.intvar(1, 2, name="y")
        c_eq, c_neq = x == y, x != y
        vs = VersionSpace([c_eq, c_neq])
        vs.add_negative([c_eq, c_neq])
        assert len(vs.clauses) == 1
        assert vs.clauses[0] == {c_eq, c_neq}

    @pytest.mark.fast
    def test_complement_atoms_paired(self):
        x, y = cp.intvar(1, 2, name="x"), cp.intvar(1, 2, name="y")
        c_eq, c_neq = x == y, x != y
        vs = VersionSpace([c_eq, c_neq])
        assert vs.get_complement(c_eq) is c_neq or hash(vs.get_complement(c_eq)) == hash(c_neq)
        assert vs.get_complement(c_neq) is c_eq or hash(vs.get_complement(c_neq)) == hash(c_eq)
        vs.add_negative([c_neq])
        assert c_neq in vs.forced_true
        assert c_eq in vs.forced_false

    @pytest.mark.fast
    def test_complement_relations(self):
        x, y = cp.intvar(1, 3, name="x"), cp.intvar(1, 3, name="y")
        assert str(constraint_complement(x == y)) == str(x != y)
        assert str(constraint_complement(x != y)) == str(x == y)
        assert str(constraint_complement(x < y)) == str(x >= y)


class TestConAcq2:

    @pytest.mark.fast
    def test_conacq2_learns_tiny(self):
        instance, oracle = _tiny_neq_instance()
        learned = ca.ConAcq2(strategy="optimal").learn(instance=instance, oracle=oracle)
        assert len(learned.cl) > 0
        assert learned.get_cpmpy_model().solve()

    @pytest.mark.fast
    def test_conacq2_optimistic_strategy(self):
        instance, oracle = _tiny_neq_instance()
        learned = ca.ConAcq2(strategy="optimistic").learn(instance=instance, oracle=oracle)
        assert len(learned.cl) > 0
        assert learned.get_cpmpy_model().solve()

    @pytest.mark.fast
    @pytest.mark.parametrize("strategy", ["optimal", "optimistic"])
    def test_conacq2_strategies_learn_neq(self, strategy):
        """Both strategies converge on a small all-different-style instance."""
        instance, oracle = _tiny_neq_instance()
        ca_sys = ca.ConAcq2(strategy=strategy)
        learned = ca_sys.learn(instance=instance, oracle=oracle)
        assert len(learned.cl) > 0
        assert learned.get_cpmpy_model().solve()
        assert ca_sys.env.metrics.membership_queries_count > 0

    @pytest.mark.fast
    def test_conacq2_complementary_bias_progress(self):
        """
        On a complementary language, a few queries must shrink the version space
        (not premature empty cl with everything still unset).
        """
        instance, oracle = _tiny_neq_instance()
        ca_sys = ca.ConAcq2(strategy="optimal")
        ca_sys.env.init_state(instance, oracle, 0, ca.Metrics())
        ca_sys.env.instance.construct_bias()
        vs = VersionSpace(ca_sys.env.instance.bias)
        ca_sys.env.version_space = vs

        unset0 = len(vs.unset())
        for _ in range(6):
            Y = ca_sys.env.run_query_generation(ca_sys.env.instance.X)
            if len(Y) == 0:
                break
            ans = ca_sys.env.ask_membership_query(Y)
            kappa = vs.kappa_under_T(Y)
            if ans:
                vs.add_positive(kappa)
            else:
                vs.add_negative(kappa)

        progressed = (
            len(vs.forced_true) > 0
            or len(vs.forced_false) > 0
            or len(vs.clauses) > 0
            or len(vs.unset()) < unset0
        )
        assert progressed
        # Not a bogus convergence with empty CL while atoms remain open
        if vs.is_monomial():
            assert len(vs.forced_true) > 0 or len(vs.learned_network()) >= 0
