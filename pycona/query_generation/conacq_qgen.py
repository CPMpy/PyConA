import cpmpy as cp
from cpmpy.solvers.solver_interface import ExitStatus

from .qgen_core import QGenBase
from ..ca_environment.active_ca import ActiveCAEnv
from ..active_algorithms.conacq2 import constraint_complement


class ConAcqQGen(QGenBase):
    """
    Query generation for ConAcq.2 (Algorithms 5–7), LIRMM-style dual solve:

    BuildFormula over Boolean atoms a(c)/a(c-bar) → candidate network φ(I) →
    CP solveQ for a complete assignment. CP-unsat networks yield nogoods N.
    """

    def __init__(self, ca_env: ActiveCAEnv = None, *, strategy="optimal", time_limit=2):
        """
        :param ca_env: The CA environment.
        :param strategy: ``\"optimal\"`` or ``\"optimistic\"``.
        :param time_limit: Time limit for Boolean / CP solves.
        """
        super().__init__(ca_env, time_limit)
        assert strategy in ("optimal", "optimistic")
        self._strategy = strategy

    @property
    def strategy(self):
        """Query generation strategy."""
        return self._strategy

    @strategy.setter
    def strategy(self, strategy):
        assert strategy in ("optimal", "optimistic")
        self._strategy = strategy

    def generate(self, Y=None):
        """
        Generate a complete irredundant membership query.

        :param Y: Variables to assign (default: all instance variables).
        :return: List of variables forming the query, or empty set on convergence.
        """
        if Y is None:
            Y = self.env.instance.X
        assert isinstance(Y, list), "When generating a query, Y must be a list of variables"

        vs = getattr(self.env, "version_space", None)
        if vs is None:
            raise RuntimeError("ConAcqQGen requires env.version_space to be set by ConAcq2")

        if vs.is_monomial():
            q = self._irredundant_query_monomial(vs, Y)
            return q if q is not None else set()

        q = self._query_gen(vs, Y)
        if q is not None:
            return q

        q = self._irredundant_query_monomial(vs, Y)
        return q if q is not None else set()

    def _target_t(self, alpha):
        if self.strategy == "optimal":
            return 1
        if alpha:
            return max(1, len(alpha) - 1)
        return 1

    def _query_gen(self, vs, Y):
        """
        Algorithm 5: BuildFormula → SAT model → solveQ, with nogood learning.
        """
        q = None
        alpha = None
        epsilon = 0
        skip_rebuild = False
        formula_cons = None
        max_iters = max(200, 4 * len(vs.bias) + 20)
        iters = 0

        while q is None and not vs.is_monomial() and iters < max_iters:
            iters += 1

            if not skip_rebuild:
                if alpha is None:
                    candidates = vs.non_unary_unmarked_clauses()
                    if candidates:
                        alpha = set(candidates[0])
                        epsilon = 0

                t = self._target_t(alpha)
                alpha_list = list(alpha) if alpha else []
                splittable = bool(alpha_list) and (
                    (t + epsilon < len(alpha_list)) or (t - epsilon > 0)
                )
                if alpha_list and epsilon > len(alpha_list):
                    splittable = False

                formula_cons = self._build_formula(vs, alpha_list, t, epsilon, splittable)
            else:
                # Only N changed; rebuild formula with same alpha/t/ε
                t = self._target_t(alpha)
                alpha_list = list(alpha) if alpha else []
                splittable = bool(alpha_list) and (
                    (t + epsilon < len(alpha_list)) or (t - epsilon > 0)
                )
                formula_cons = self._build_formula(vs, alpha_list, t, epsilon, splittable)
                skip_rebuild = False

            model_I = self._solve_boolean(formula_cons, minimize_bias_atoms=vs if not alpha_list else None)
            if model_I is None:
                # Timeout / solver error: stop generating
                return None

            if model_I is False:
                # Unsat formula
                if splittable:
                    epsilon += 1
                else:
                    if alpha:
                        vs.unitize_clause(alpha)
                    alpha = None
                    epsilon = 0
                continue

            network = self._network_from_model(vs, model_I)
            sol = self._solve_q(network, Y)
            if sol is True:
                if (not splittable) and alpha:
                    vs.mark_clause(alpha)
                alpha = None
                epsilon = 0
                q = list(Y)
                break

            if sol is None:
                # CP timeout — abort this generation attempt
                return None

            # CP unsat: learn nogood and retry (LIRMM skip)
            conflict = self._conflict_set(network, Y)
            if conflict:
                vs.add_nogood(conflict)
                if alpha:
                    skip_rebuild = True
                else:
                    # Avoid infinite loop on empty alpha: diversify by clearing
                    pass
            else:
                # Cannot extract conflict; give up on this alpha
                if alpha:
                    vs.mark_clause(alpha)
                alpha = None
                epsilon = 0

        return q

    def _build_formula(self, vs, alpha, t, epsilon, splittable):
        """
        Algorithm 6: Boolean formula whose models are candidate networks φ(I).

        :return: List of CPMpy Boolean constraints.
        """
        cons = []
        cons.extend(vs.boolean_units())
        cons.extend(vs.boolean_clauses())
        cons.extend(vs.boolean_nogoods())

        alpha_set = set(alpha)
        alpha_bars = set()
        for c in alpha:
            cbar = vs.get_complement(c)
            if cbar is not None:
                alpha_bars.add(cbar)

        if alpha:
            # For each bias constraint with an atom
            for c in vs.bias:
                if c not in vs.atom:
                    continue
                cbar = vs.get_complement(c)
                if splittable and c not in alpha_set and (cbar is None or cbar not in alpha_set):
                    # Force outside α into the network
                    if vs.is_unset(c):
                        cons.append(vs.atom[c])
                if c in alpha_set and cbar is not None and cbar in vs.atom:
                    cons.append(cp.any([vs.atom[c], vs.atom[cbar]]))

            # Cardinality on how many of α stay true
            alpha_atoms = [vs.atom[c] for c in alpha if c in vs.atom]
            n = len(alpha_atoms)
            if n == 0:
                return cons
            if splittable:
                lower = max(n - t - epsilon, 1)
                upper = min(n - t + epsilon, n - 1)
            else:
                lower = 1
                upper = max(n - 1, 1)
            if lower > upper:
                # Impossible bounds → force unsat
                cons.append(False)
            else:
                s = cp.sum(alpha_atoms)
                cons.append(s >= lower)
                cons.append(s <= upper)
        else:
            # At least one unset complement true; keep other unset bias atoms false
            # so φ(I) stays sparse (assignment only needs to witness one open cut).
            to_add = []
            for c in vs.unset():
                cbar = vs.get_complement(c)
                if cbar is not None and cbar in vs.atom:
                    to_add.append(vs.atom[cbar])
                if c in vs.atom and vs.is_unset(c):
                    # Do not force a(c) true when we only need a complement witness
                    pass
            if to_add:
                cons.append(cp.any(to_add) if len(to_add) > 1 else to_add[0])
            else:
                unset_atoms = [vs.atom[c] for c in vs.unset() if c in vs.atom]
                if unset_atoms:
                    cons.append(cp.any(unset_atoms) if len(unset_atoms) > 1 else unset_atoms[0])
                else:
                    cons.append(False)

        return cons

    def _solve_boolean(self, cons, minimize_bias_atoms=None):
        """
        Solve the Boolean BuildFormula.

        :param minimize_bias_atoms: Optional VersionSpace; minimize number of true bias atoms.
        :return: dict atom→bool on success, False if unsat, None on timeout/error.
        """
        if any(c is False for c in cons):
            return False
        cons = [c for c in cons if c is not False and c is not True]
        if not cons:
            return {}
        m = cp.Model(cons)
        try:
            s = cp.SolverLookup.get("ortools", m)
            if minimize_bias_atoms is not None:
                bias_atoms = [minimize_bias_atoms.atom[c]
                              for c in minimize_bias_atoms.bias
                              if c in minimize_bias_atoms.atom]
                if bias_atoms:
                    s.minimize(cp.sum(bias_atoms))
            flag = s.solve(time_limit=self.time_limit)
        except Exception:
            return None
        if not flag:
            status = getattr(s, "cpm_status", None)
            if status is not None and status.exitstatus == ExitStatus.UNSATISFIABLE:
                return False
            return None

        from cpmpy.transformations.get_variables import get_variables
        vals = {}
        for v in get_variables(cons):
            if v.value() is not None:
                vals[v] = bool(v.value())
        return vals

    def _network_from_model(self, vs, model_I):
        """
        Build φ(I) = constraints whose atoms are true (bias members only for CP).

        Forced-true constraints are always included.
        """
        network = list(vs.forced_true)
        seen = set(network)
        for c in vs.bias:
            if c in seen:
                continue
            atom = vs.atom.get(c)
            if atom is None:
                continue
            # Prefer value from solve; fall back to atom.value()
            val = model_I.get(atom, atom.value())
            if val:
                network.append(c)
                seen.add(c)
        return network

    def _solve_q(self, network, Y):
        """
        Find a complete assignment of network.

        :return: True if solved, False if unsat, None on timeout/error.
        """
        cons = list(network) + [v >= v.lb for v in Y]
        m = cp.Model(cons)
        try:
            s = cp.SolverLookup.get("ortools", m)
            flag = s.solve(time_limit=self.time_limit)
        except Exception:
            return None
        if flag:
            if any(v.value() is None for v in Y):
                return False
            return True
        status = getattr(s, "cpm_status", None)
        if status is not None and status.exitstatus == ExitStatus.UNSATISFIABLE:
            return False
        return None

    def _conflict_set(self, network, Y):
        """
        Extract a small conflict from an unsatisfiable network via iterative deletion.

        :return: List of conflicting bias constraints, or empty if none found.
        """
        candidates = list(network)
        if not candidates:
            return []

        # Large networks: add a coarse nogood first (paper allows incomplete ConflictSets)
        if len(candidates) > 80:
            return candidates[:80]

        if self._solve_q(candidates, Y) is not False:
            return list(candidates)

        core = list(candidates)
        i = 0
        while i < len(core) and len(core) > 1:
            trial = core[:i] + core[i + 1:]
            res = self._solve_q(trial, Y)
            if res is False:
                core = trial
            else:
                i += 1
        return core

    def _irredundant_query_monomial(self, vs, Y):
        """
        Algorithm 7: IrredundantQuery when T is a monomial (or fallback).

        :return: List of variables if an irredundant query exists, else None.
        """
        cl = [c for c in vs.forced_true if c in vs.bias]
        open_cons = vs.unset()
        if not open_cons:
            return None

        m = cp.Model(list(cl) + [v >= v.lb for v in Y] + [~cp.all(open_cons)])
        try:
            s = cp.SolverLookup.get("ortools", m)
            if s.solve(time_limit=self.time_limit) and self._is_irredundant(vs, Y):
                return list(Y)
        except Exception:
            pass

        for c in list(vs.unset()):
            try:
                complement = constraint_complement(c)
            except ValueError:
                complement = ~c

            m = cp.Model(list(cl) + [complement] + [v >= v.lb for v in Y])
            try:
                s = cp.SolverLookup.get("ortools", m)
                flag = s.solve(time_limit=self.time_limit)
                unsat = (not flag) and getattr(s, "cpm_status", None) is not None \
                    and s.cpm_status.exitstatus == ExitStatus.UNSATISFIABLE
            except Exception:
                flag = False
                unsat = False

            if flag and not any(v.value() is None for v in Y) and self._is_irredundant(vs, Y):
                return list(Y)
            if unsat:
                vs.force_true(c)
                vs.propagate()
                cl = [c2 for c2 in vs.forced_true if c2 in vs.bias]

        return None

    def _is_irredundant(self, vs, Y):
        """kappa[T] nonempty and does not cover any clause of T."""
        kappa = set(vs.kappa_under_T(Y))
        if not kappa:
            return False
        for clause in vs.clauses:
            reduced = vs._reduce_clause(clause)
            if reduced and reduced.issubset(kappa):
                return False
        return True
