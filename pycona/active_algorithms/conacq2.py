import time

import cpmpy as cp
from cpmpy.expressions.core import Comparison

from .algorithm_core import AlgorithmCAInteractive
from ..problem_instance import ProblemInstance
from ..answering_queries import Oracle, UserOracle
from ..ca_environment.active_ca import ActiveCAEnv
from ..utils import get_kappa
from .. import Metrics


def constraint_complement(c):
    """
    Complement relation c-bar for common binary comparisons (needed by ConAcq.2).

    :param c: A comparison constraint.
    :return: The complementary comparison.
    """
    if not isinstance(c, Comparison):
        raise ValueError(f"No complement for non-comparison constraint: {c}")
    op, a, b = c.name, c.args[0], c.args[1]
    if op == "==":
        return a != b
    if op == "!=":
        return a == b
    if op == "<":
        return a >= b
    if op == "<=":
        return a > b
    if op == ">":
        return a <= b
    if op == ">=":
        return a < b
    raise ValueError(f"No complement for relation '{op}': {c}")


class VersionSpace:
    """
    Clausal version space for ConAcq.2.

    Bias constraints get Boolean atoms a(c). Positive examples force ~a(c);
    negatives add a clause over kappa. Complements are paired for BuildFormula.
    """

    def __init__(self, bias, background_knowledge=None):
        """
        :param bias: Candidate constraints B.
        :param background_knowledge: Optional known clauses or constraints in C_L.
        """
        self.bias = list(bias)
        self.forced_true = set()
        self.forced_false = set()
        self.clauses = []   # positive clauses: at least one c in the set is in C_T
        self.nogoods = []   # sets that cannot all be true together
        self.marked = set()

        self.atom = {}        # c -> boolvar a(c)
        self.complement = {}  # c -> c-bar (bias member when possible)
        self._init_atoms()

        if background_knowledge:
            for item in background_knowledge:
                if isinstance(item, (set, frozenset, list, tuple)):
                    self.clauses.append(set(item))
                else:
                    self.force_true(item)
            self.propagate()

    def _init_atoms(self):
        """Create a(c) for each bias constraint and pair complements."""
        by_hash = {}
        for i, c in enumerate(self.bias):
            self.atom[c] = cp.boolvar(name=f"a_{i}")
            by_hash[hash(c)] = c

        for c in self.bias:
            try:
                cbar = constraint_complement(c)
            except ValueError:
                continue
            if hash(cbar) in by_hash:
                self.complement[c] = by_hash[hash(cbar)]
            else:
                # c-bar not in B: still need an atom for BuildFormula
                self.atom[cbar] = cp.boolvar(name=f"a_bar_{len(self.atom)}")
                self.complement[c] = cbar

    def get_complement(self, c):
        return self.complement.get(c)

    def unset(self):
        """Constraints in B that are not yet decided."""
        return [c for c in self.bias
                if c not in self.forced_true and c not in self.forced_false]

    def is_unset(self, c):
        return c not in self.forced_true and c not in self.forced_false

    def is_monomial(self):
        return len(self.unset()) == 0

    def kappa_under_T(self, Y):
        """kappa of constraints still possible under T (not forced false)."""
        candidates = [c for c in self.bias if c not in self.forced_false]
        return get_kappa(candidates, Y)

    def non_unary_unmarked_clauses(self):
        result = []
        for clause in self.clauses:
            reduced = self._reduce_clause(clause)
            if len(reduced) > 1 and frozenset(reduced) not in self.marked:
                result.append(reduced)
        return result

    def mark_clause(self, clause):
        self.marked.add(frozenset(self._reduce_clause(clause)))

    def unitize_clause(self, clause):
        """All literals of alpha must be true; drop the clause (non-splittable)."""
        alpha = self._reduce_clause(clause)
        for c in alpha:
            self.force_true(c)
        key = frozenset(alpha)
        self.clauses = [cl for cl in self.clauses
                        if frozenset(self._reduce_clause(cl)) != key]
        self.propagate()

    def add_positive(self, kappa):
        """Positive example: ~a(c) for each c in kappa."""
        for c in kappa:
            self.force_false(c)
        self.propagate()

    def add_negative(self, kappa):
        """Negative example: OR a(c) over kappa (with simple subsumption)."""
        clause = set(kappa) - self.forced_false
        if not clause or (clause & self.forced_true):
            return

        kept = []
        subsumed = False
        for existing in self.clauses:
            reduced = self._reduce_clause(existing)
            if not reduced:
                continue
            if clause.issubset(reduced):
                continue  # new clause is stronger
            if reduced.issubset(clause):
                subsumed = True  # existing is stronger
            kept.append(reduced)

        if subsumed:
            self.clauses = kept
            self.propagate()
            return

        if len(clause) == 1:
            c = next(iter(clause))
            self.force_true(c)
            cbar = self.get_complement(c)
            if cbar is not None:
                self.force_false(cbar)
            self.clauses = kept
        else:
            kept.append(clause)
            self.clauses = kept
        self.propagate()

    def add_nogood(self, conflict_set):
        """Not all constraints in the conflict can be true together."""
        cs = set(conflict_set) - self.forced_false
        if not cs:
            return
        if len(cs) == 1:
            self.force_false(next(iter(cs)))
        elif frozenset(cs) not in {frozenset(ng) for ng in self.nogoods}:
            self.nogoods.append(cs)
        self.propagate()

    def force_true(self, c):
        if c in self.forced_false:
            return
        self.forced_true.add(c)
        cbar = self.get_complement(c)
        if cbar is not None:
            self.forced_false.add(cbar)

    def force_false(self, c):
        if c in self.forced_true:
            return
        self.forced_false.add(c)

    def _reduce_clause(self, clause):
        if set(clause) & self.forced_true:
            return set()
        return set(clause) - self.forced_false

    def propagate(self):
        """Unit propagation on clauses and nogoods."""
        changed = True
        while changed:
            changed = False

            new_clauses = []
            for clause in self.clauses:
                reduced = self._reduce_clause(clause)
                if not reduced:
                    continue
                if len(reduced) == 1:
                    c = next(iter(reduced))
                    if c not in self.forced_true:
                        self.force_true(c)
                        changed = True
                else:
                    new_clauses.append(reduced)
            self.clauses = new_clauses

            new_nogoods = []
            for ng in self.nogoods:
                remaining = set(ng) - self.forced_false
                if not remaining:
                    continue
                undecided = remaining - self.forced_true
                if len(undecided) == 1:
                    c = next(iter(undecided))
                    if c not in self.forced_false:
                        self.force_false(c)
                        changed = True
                elif undecided:
                    new_nogoods.append(remaining)
            self.nogoods = new_nogoods

    def bool_encoding(self):
        """CPMpy encoding of forced literals, clauses and nogoods over a(c)."""
        cons = []
        for c in self.forced_true:
            if c in self.atom:
                cons.append(self.atom[c])
        for c in self.forced_false:
            if c in self.atom:
                cons.append(~self.atom[c])
        for clause in self.clauses:
            reduced = self._reduce_clause(clause)
            lits = [self.atom[c] for c in reduced if c in self.atom]
            if len(lits) == 1:
                cons.append(lits[0])
            elif len(lits) > 1:
                cons.append(cp.any(lits))
        for ng in self.nogoods:
            remaining = set(ng) - self.forced_false
            lits = [~self.atom[c] for c in remaining if c in self.atom]
            if len(lits) == 1:
                cons.append(lits[0])
            elif len(lits) > 1:
                cons.append(cp.any(lits))
        return cons

    def learned_network(self):
        return [c for c in self.forced_true if c in self.bias]


class ConAcq2(AlgorithmCAInteractive):
    """
    ConAcq.2: complete membership queries + clausal version space
    (Bessiere et al., AIJ 2017).
    """

    def __init__(self, ca_env: ActiveCAEnv = None, *, strategy="optimal", background_knowledge=None):
        """
        :param ca_env: An instance of ActiveCAEnv, default is None.
        :param strategy: "optimal" (t=1) or "optimistic".
        :param background_knowledge: Optional prior knowledge for the version space.
        """
        assert strategy in ("optimal", "optimistic"), \
            "strategy must be 'optimal' or 'optimistic'"
        self.strategy = strategy
        self.background_knowledge = background_knowledge

        from ..query_generation.conacq_qgen import ConAcqQGen
        if ca_env is None:
            ca_env = ActiveCAEnv(qgen=ConAcqQGen(strategy=strategy))
        super().__init__(ca_env)
        if not isinstance(self.env.qgen, ConAcqQGen):
            self.env.qgen = ConAcqQGen(strategy=strategy)
        else:
            self.env.qgen.strategy = strategy

    def learn(self, instance: ProblemInstance, oracle: Oracle = UserOracle(), verbose=0, X=None,
              metrics: Metrics = None):
        """
        Learn constraints using ConAcq.2.

        :param instance: the problem instance to acquire the constraints for
        :param oracle: An instance of Oracle, default is to use the user as the oracle.
        :param verbose: Verbosity level, default is 0.
        :param X: The set of variables to consider, default is None.
        :param metrics: statistics logger during learning
        :return: the learned instance
        """
        if X is None:
            X = instance.X
        assert isinstance(X, list), \
            "When using .learn(), set parameter X must be a list of variables. Instead got: {}".format(X)
        assert set(X).issubset(set(instance.X)), \
            "When using .learn(), set parameter X must be a subset of the problem instance variables. Instead got: {}".format(X)

        self.env.init_state(instance, oracle, verbose, metrics)

        if len(self.env.instance.bias) == 0:
            self.env.instance.construct_bias(X)

        bg = list(self.background_knowledge) if self.background_knowledge else []
        bg = bg + list(self.env.instance.cl)

        vs = VersionSpace(self.env.instance.bias, background_knowledge=bg)
        self.env.version_space = vs
        self.env.qgen.strategy = self.strategy
        self._sync_instance_from_vs(vs)

        while True:
            if self.env.verbose > 2:
                print("Size of CL: ", len(self.env.instance.cl))
                print("Size of B: ", len(self.env.instance.bias))
                print("Unset atoms: ", len(vs.unset()))
                print("Number of Queries: ", self.env.metrics.membership_queries_count)

            gen_start = time.time()
            Y = self.env.run_query_generation(X)
            gen_end = time.time()

            if len(Y) == 0:
                # no query -> converged
                self._sync_instance_from_vs(vs)
                self.env.metrics.finalize_statistics()
                self.env.converged = True
                if self.env.verbose >= 1:
                    print(f"\nLearned {self.env.metrics.cl} constraints in "
                          f"{self.env.metrics.membership_queries_count} queries.")
                return self.env.instance

            self.env.metrics.increase_generation_time(gen_end - gen_start)
            self.env.metrics.increase_generated_queries()
            self.env.metrics.increase_top_queries()

            kappa = vs.kappa_under_T(Y)
            answer = self.env.ask_membership_query(Y)
            if answer:
                vs.add_positive(kappa)
            else:
                vs.add_negative(kappa)

            self._sync_instance_from_vs(vs)

    def _sync_instance_from_vs(self, vs: VersionSpace):
        """Keep instance.cl / bias in sync with forced literals."""
        cl_set = set(self.env.instance.cl)
        bias_set = set(self.env.instance.bias)

        to_add = [c for c in vs.learned_network() if c not in cl_set]
        in_bias = [c for c in to_add if c in bias_set]
        not_in_bias = [c for c in to_add if c not in bias_set]
        if in_bias:
            self.env.add_to_cl(in_bias)
        if not_in_bias:
            self.env.instance.cl.extend(not_in_bias)
            self.env.metrics.cl += len(not_in_bias)

        to_remove = [c for c in vs.forced_false if c in bias_set]
        if to_remove:
            self.env.remove_from_bias(to_remove)
