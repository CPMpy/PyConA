import time

import cpmpy as cp

from .algorithm_core import AlgorithmCAInteractive
from ..problem_instance import ProblemInstance
from ..answering_queries import Oracle, UserOracle
from ..ca_environment.active_ca import ActiveCAEnv
from ..utils import get_kappa
from .. import Metrics


def constraint_complement(c):
    """
    Return the complement constraint c-bar for common binary comparisons.

    :param c: A CPMpy comparison constraint.
    :return: The complement constraint.
    :raises ValueError: If the relation has no supported complement.
    """
    from cpmpy.expressions.core import Comparison
    if not isinstance(c, Comparison):
        raise ValueError(f"No complement available for non-comparison constraint: {c}")
    op = c.name
    a, b = c.args[0], c.args[1]
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
    raise ValueError(f"No complement available for relation '{op}' in constraint: {c}")


def _expr_key(c):
    """Stable-ish key for matching complement expressions already in the bias."""
    return str(c)


class VersionSpace:
    """
    Clausal encoding of the learner's version space for ConAcq.2.

    Atoms a(c) (CPMpy boolvars) mean "constraint c belongs to the target network".
    Complements c-bar have paired atoms for BuildFormula.
    """

    def __init__(self, bias, background_knowledge=None):
        """
        :param bias: Candidate constraints B.
        :param background_knowledge: Optional prior clauses or constraints forced true.
        """
        self.bias = list(bias)
        self.forced_true = set()
        self.forced_false = set()
        self.clauses = []
        # Nogoods: not all constraints in the set can be true together
        self.nogoods = []
        self.marked = set()

        self.atom = {}
        self.complement = {}          # c -> complement expression (bias member or shadow)
        self.complement_of = {}       # shadow complement -> original c
        self._shadow_exprs = {}       # key -> shadow complement expression not in bias

        self._init_atoms()

        if background_knowledge:
            for item in background_knowledge:
                if isinstance(item, (set, frozenset, list, tuple)):
                    self.clauses.append(set(item))
                else:
                    self.force_true(item)
            self.propagate()

    def _init_atoms(self):
        """Create a(c) for each bias member and pair complements / shadow atoms."""
        for i, c in enumerate(self.bias):
            self.atom[c] = cp.boolvar(name=f"a_{i}")

        bias_set = set(self.bias)
        for c in list(self.bias):
            try:
                cbar_expr = constraint_complement(c)
            except ValueError:
                continue
            # Prefer the existing bias object when complement is already in B
            matched = None
            if cbar_expr in bias_set:
                matched = cbar_expr
            else:
                h = hash(cbar_expr)
                for b in self.bias:
                    if hash(b) == h:
                        matched = b
                        break
            if matched is not None:
                self.complement[c] = matched
            else:
                key = _expr_key(cbar_expr)
                if key not in self._shadow_exprs:
                    self._shadow_exprs[key] = cbar_expr
                    self.atom[cbar_expr] = cp.boolvar(name=f"a_bar_{len(self.atom)}")
                    self.complement_of[cbar_expr] = c
                self.complement[c] = self._shadow_exprs[key]

    def atom_of(self, c):
        """Return the Boolean atom for constraint c (bias or shadow)."""
        return self.atom[c]

    def has_complement(self, c):
        return c in self.complement

    def get_complement(self, c):
        return self.complement.get(c)

    def is_bias_constraint(self, c):
        """True if c is a member of the acquisition bias (not a shadow complement)."""
        return c in self.bias

    def unset(self):
        """Bias constraints whose membership is not yet decided."""
        return [c for c in self.bias
                if c not in self.forced_true and c not in self.forced_false]

    def is_unset(self, c):
        return c not in self.forced_true and c not in self.forced_false

    def is_monomial(self):
        """True when every bias atom is fixed."""
        return len(self.unset()) == 0

    def kappa_under_T(self, Y):
        """
        Constraints not ruled out by T that reject the current assignment on Y.

        :param Y: Variables involved in the query.
        :return: List of violated constraints still possible under T.
        """
        candidates = [c for c in self.bias if c not in self.forced_false]
        return get_kappa(candidates, Y)

    def non_unary_unmarked_clauses(self):
        """Positive clauses with size > 1 that are not marked as non-splittable."""
        result = []
        for clause in self.clauses:
            reduced = self._reduce_clause(clause)
            if len(reduced) > 1 and frozenset(reduced) not in self.marked:
                result.append(reduced)
        return result

    def mark_clause(self, clause):
        """Mark a clause as non-splittable."""
        self.marked.add(frozenset(self._reduce_clause(clause)))

    def unitize_clause(self, clause):
        """Force all literals of alpha true and remove the clause (non-splittable)."""
        alpha = self._reduce_clause(clause)
        for c in alpha:
            self.force_true(c)
        key = frozenset(alpha)
        self.clauses = [cl for cl in self.clauses
                        if frozenset(self._reduce_clause(cl)) != key]
        self.propagate()

    def add_positive(self, kappa):
        """
        Process a positive example: add ~a(c) for each c in kappa.

        :param kappa: Constraints violated by the positive example.
        """
        for c in kappa:
            self.force_false(c)
        self.propagate()

    def add_negative(self, kappa):
        """
        Process a negative example: add OR{a(c) | c in kappa}.

        :param kappa: Constraints violated by the negative example.
        """
        clause = set(kappa) - self.forced_false
        if not clause:
            return
        if clause & self.forced_true:
            return

        new_clauses = []
        subsumed_by_existing = False
        for existing in self.clauses:
            reduced = self._reduce_clause(existing)
            if not reduced:
                continue
            if clause.issubset(reduced):
                continue
            if reduced.issubset(clause):
                subsumed_by_existing = True
            new_clauses.append(reduced)

        if subsumed_by_existing:
            self.clauses = new_clauses
            self.propagate()
            return

        if len(clause) == 1:
            c = next(iter(clause))
            self.force_true(c)
            # LIRMM: also rule out the complement
            cbar = self.get_complement(c)
            if cbar is not None:
                self.force_false(cbar)
            self.clauses = new_clauses
        else:
            new_clauses.append(clause)
            self.clauses = new_clauses
        self.propagate()

    def add_nogood(self, conflict_set):
        """
        Record that the constraints in conflict_set cannot all belong to a network.

        :param conflict_set: Iterable of constraints forming a conflict.
        """
        cs = set(conflict_set) - self.forced_false
        if not cs:
            return
        if len(cs) == 1:
            self.force_false(next(iter(cs)))
        else:
            # Avoid duplicate nogoods
            fs = frozenset(cs)
            if not any(frozenset(ng) == fs for ng in self.nogoods):
                self.nogoods.append(cs)
        self.propagate()

    def force_true(self, c):
        """Force a(c) to true; rule out paired complement when known."""
        if c in self.forced_false:
            return
        self.forced_true.add(c)
        cbar = self.get_complement(c)
        if cbar is not None and cbar not in self.forced_true:
            self.forced_false.add(cbar)

    def force_false(self, c):
        """Force a(c) to false."""
        if c in self.forced_true:
            return
        self.forced_false.add(c)

    def _reduce_clause(self, clause):
        """Drop literals already forced false; empty if already satisfied."""
        if set(clause) & self.forced_true:
            return set()
        return set(clause) - self.forced_false

    def propagate(self):
        """Unit-propagate clauses and nogoods until fixpoint."""
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
                forced_in = remaining & self.forced_true
                undecided = remaining - self.forced_true
                if not undecided:
                    continue
                if len(undecided) == 1 and (forced_in == remaining - undecided or len(remaining) == 1):
                    c = next(iter(undecided))
                    if c not in self.forced_false:
                        self.force_false(c)
                        changed = True
                else:
                    new_nogoods.append(remaining)
            self.nogoods = new_nogoods

    def boolean_units(self):
        """
        CPMpy constraints encoding forced literals over atoms.

        :return: List of Boolean constraints.
        """
        cons = []
        for c in self.forced_true:
            if c in self.atom:
                cons.append(self.atom[c])
        for c in self.forced_false:
            if c in self.atom:
                cons.append(~self.atom[c])
        return cons

    def boolean_clauses(self):
        """CPMpy encoding of positive clauses of T."""
        cons = []
        for clause in self.clauses:
            reduced = self._reduce_clause(clause)
            if not reduced:
                continue
            lits = [self.atom[c] for c in reduced if c in self.atom]
            if len(lits) == 1:
                cons.append(lits[0])
            elif len(lits) > 1:
                cons.append(cp.any(lits))
        return cons

    def boolean_nogoods(self):
        """CPMpy encoding of nogoods N: not all atoms true."""
        cons = []
        for ng in self.nogoods:
            remaining = set(ng) - self.forced_false
            lits = [~self.atom[c] for c in remaining if c in self.atom]
            if len(lits) == 1:
                cons.append(lits[0])
            elif len(lits) > 1:
                cons.append(cp.any(lits))
        return cons

    def learned_network(self):
        """Constraints forced to belong to the target network."""
        return [c for c in self.forced_true if c in self.bias]


class ConAcq2(AlgorithmCAInteractive):
    """
    ConAcq.2 active constraint acquisition using membership queries and a clausal
    version space.

    Based on:
    Bessiere, C., et al. (2017). Constraint acquisition. Artificial Intelligence, 244, 315-342.
    """

    def __init__(self, ca_env: ActiveCAEnv = None, *, strategy="optimal", background_knowledge=None):
        """
        Initialize ConAcq.2.

        :param ca_env: An instance of ActiveCAEnv, default is None.
        :param strategy: Query generation strategy, ``\"optimal\"`` or ``\"optimistic\"``.
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
        Learn constraints with ConAcq.2 (complete membership queries + version space).

        :param instance: the problem instance to acquire the constraints for
        :param oracle: An instance of Oracle, default is to use the user as the oracle.
        :param verbose: Verbosity level, default is 0.
        :param X: The set of variables to consider, default is None (all variables).
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
        bg = list(bg) + list(self.env.instance.cl)

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

            answer = self.env.ask_membership_query(Y)
            kappa = vs.kappa_under_T(Y)

            if answer:
                vs.add_positive(kappa)
            else:
                vs.add_negative(kappa)

            self._sync_instance_from_vs(vs)

    def _sync_instance_from_vs(self, vs: VersionSpace):
        """Mirror forced literals into instance.cl / instance.bias for PyConA compatibility."""
        cl_set = set(self.env.instance.cl)
        to_add = [c for c in vs.learned_network() if c not in cl_set]
        if to_add:
            already_in_bias = [c for c in to_add if c in set(self.env.instance.bias)]
            missing_from_bias = [c for c in to_add if c not in set(self.env.instance.bias)]
            if already_in_bias:
                self.env.add_to_cl(already_in_bias)
            if missing_from_bias:
                self.env.instance.cl.extend(missing_from_bias)
                self.env.metrics.cl += len(missing_from_bias)

        to_remove = [c for c in vs.forced_false if c in set(self.env.instance.bias)]
        if to_remove:
            self.env.remove_from_bias(to_remove)
