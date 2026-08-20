import cpmpy as cp
from cpmpy.solvers.solver_interface import ExitStatus
from cpmpy.transformations.get_variables import get_variables

from .qgen_core import QGenBase
from ..ca_environment.active_ca import ActiveCAEnv
from ..active_algorithms.conacq2 import constraint_complement


class ConAcqQGen(QGenBase):
    """
    Query generation for ConAcq.2 (Algorithms 5-7).

    BuildFormula over a(c) -> Boolean model as network -> CP solveQ.
    Unsat networks are recorded as nogoods.
    """

    def __init__(self, ca_env: ActiveCAEnv = None, *, strategy="optimal", time_limit=2):
        """
        :param ca_env: The CA environment.
        :param strategy: "optimal" or "optimistic".
        :param time_limit: Time limit for Boolean / CP solves.
        """
        super().__init__(ca_env, time_limit)
        assert strategy in ("optimal", "optimistic")
        self._strategy = strategy

    @property
    def strategy(self):
        return self._strategy

    @strategy.setter
    def strategy(self, strategy):
        assert strategy in ("optimal", "optimistic")
        self._strategy = strategy

    def generate(self, Y=None):
        """
        Generate a complete membership query, or empty set on convergence.

        :param Y: Variables to assign (default: all instance variables).
        :return: List of variables, or empty set.
        """
        if Y is None:
            Y = self.env.instance.X
        assert isinstance(Y, list), "When generating a query, Y must be a list of variables"

        vs = getattr(self.env, "version_space", None)
        if vs is None:
            raise RuntimeError("ConAcqQGen requires env.version_space (set by ConAcq2)")

        if not vs.is_monomial():
            q = self._query_gen(vs, Y)
            if q is not None:
                return q

        q = self._irredundant_query(vs, Y)
        return q if q is not None else set()

    def _target_t(self, alpha):
        if self.strategy == "optimal":
            return 1
        return max(1, len(alpha) - 1) if alpha else 1

    def _query_gen(self, vs, Y):
        """Main loop: BuildFormula -> Boolean model -> solveQ, with nogoods."""
        alpha = None
        epsilon = 0
        max_iters = max(200, 4 * len(vs.bias) + 20)

        for _ in range(max_iters):
            if vs.is_monomial():
                return None

            if alpha is None:
                candidates = vs.non_unary_unmarked_clauses()
                if candidates:
                    alpha = set(candidates[0])
                    epsilon = 0

            alpha_list = list(alpha) if alpha else []
            t = self._target_t(alpha_list)
            splittable = bool(alpha_list) and (
                (t + epsilon < len(alpha_list)) or (t - epsilon > 0)
            )
            if alpha_list and epsilon > len(alpha_list):
                splittable = False

            formula = self._build_formula(vs, alpha_list, t, epsilon, splittable)
            # empty alpha: prefer sparse networks (fewer true bias atoms)
            model = self._solve_boolean(formula, sparse=(not alpha_list), vs=vs)

            if model is None:
                return None
            if model is False:
                # Boolean unsat: relax epsilon, or unitize if not splittable
                if splittable:
                    epsilon += 1
                else:
                    if alpha:
                        vs.unitize_clause(alpha)
                    alpha = None
                    epsilon = 0
                continue

            network = self._to_network(vs, model)
            sat = self._solve_q(network, Y)

            if sat is True:
                if alpha and not splittable:
                    vs.mark_clause(alpha)
                return list(Y)
            if sat is None:
                return None

            # CP unsat -> add nogood and retry
            conflict = self._conflict_set(network, Y)
            if conflict:
                vs.add_nogood(conflict)
            elif alpha:
                vs.mark_clause(alpha)
                alpha = None
                epsilon = 0

        return None

    def _build_formula(self, vs, alpha, t, epsilon, splittable):
        """Boolean constraints whose models are candidate networks."""
        cons = list(vs.bool_encoding())
        alpha_set = set(alpha)

        if alpha:
            for c in vs.bias:
                if c not in vs.atom:
                    continue
                cbar = vs.get_complement(c)
                # outside alpha: force a(c) when splittable
                if splittable and c not in alpha_set and (cbar is None or cbar not in alpha_set):
                    if vs.is_unset(c):
                        cons.append(vs.atom[c])
                # inside alpha: a(c) \/ a(c-bar)
                if c in alpha_set and cbar is not None and cbar in vs.atom:
                    cons.append(cp.any([vs.atom[c], vs.atom[cbar]]))

            atoms = [vs.atom[c] for c in alpha if c in vs.atom]
            n = len(atoms)
            if n == 0:
                return cons
            if splittable:
                lower = max(n - t - epsilon, 1)
                upper = min(n - t + epsilon, n - 1)
            else:
                lower, upper = 1, max(n - 1, 1)
            if lower > upper:
                cons.append(False)
            else:
                s = cp.sum(atoms)
                cons.append(s >= lower)
                cons.append(s <= upper)
        else:
            # at least one open complement (or unset atom) must be true
            bars = []
            for c in vs.unset():
                cbar = vs.get_complement(c)
                if cbar is not None and cbar in vs.atom:
                    bars.append(vs.atom[cbar])
            if bars:
                cons.append(cp.any(bars) if len(bars) > 1 else bars[0])
            else:
                unset_atoms = [vs.atom[c] for c in vs.unset() if c in vs.atom]
                if unset_atoms:
                    cons.append(cp.any(unset_atoms) if len(unset_atoms) > 1 else unset_atoms[0])
                else:
                    cons.append(False)

        return cons

    def _solve_boolean(self, cons, sparse=False, vs=None):
        """
        Solve the Boolean formula.

        :return: dict atom->bool, False if unsat, None on timeout/error.
        """
        if any(c is False for c in cons):
            return False
        cons = [c for c in cons if c is not False and c is not True]
        if not cons:
            return {}

        m = cp.Model(cons)
        try:
            s = cp.SolverLookup.get("ortools", m)
            if sparse and vs is not None:
                atoms = [vs.atom[c] for c in vs.bias if c in vs.atom]
                if atoms:
                    s.minimize(cp.sum(atoms))
            flag = s.solve(time_limit=self.time_limit)
        except Exception:
            return None

        if not flag:
            status = getattr(s, "cpm_status", None)
            if status is not None and status.exitstatus == ExitStatus.UNSATISFIABLE:
                return False
            return None

        return {v: bool(v.value()) for v in get_variables(cons) if v.value() is not None}

    def _to_network(self, vs, model):
        """phi(I): bias constraints with a(c) true, plus forced_true."""
        network = list(vs.forced_true)
        seen = set(network)
        for c in vs.bias:
            if c in seen or c not in vs.atom:
                continue
            if model.get(vs.atom[c], False):
                network.append(c)
                seen.add(c)
        return network

    def _solve_q(self, network, Y):
        """
        Find a complete assignment of the network.

        :return: True if sat, False if unsat, None on timeout/error.
        """
        m = cp.Model(list(network) + [v >= v.lb for v in Y])
        try:
            s = cp.SolverLookup.get("ortools", m)
            flag = s.solve(time_limit=self.time_limit)
        except Exception:
            return None
        if flag:
            return all(v.value() is not None for v in Y)
        status = getattr(s, "cpm_status", None)
        if status is not None and status.exitstatus == ExitStatus.UNSATISFIABLE:
            return False
        return None

    def _conflict_set(self, network, Y):
        """Small unsat core by deleting constraints one by one (coarse if large)."""
        if not network:
            return []
        if len(network) > 80:
            return list(network[:80])

        core = list(network)
        i = 0
        while i < len(core) and len(core) > 1:
            trial = core[:i] + core[i + 1:]
            if self._solve_q(trial, Y) is False:
                core = trial
            else:
                i += 1
        return core

    def _irredundant_query(self, vs, Y):
        """IrredundantQuery when T is monomial (or as fallback)."""
        cl = [c for c in vs.forced_true if c in vs.bias]
        open_cons = vs.unset()
        if not open_cons:
            return None

        m = cp.Model(cl + [v >= v.lb for v in Y] + [~cp.all(open_cons)])
        try:
            if cp.SolverLookup.get("ortools", m).solve(time_limit=self.time_limit):
                if self._is_irredundant(vs, Y):
                    return list(Y)
        except Exception:
            pass

        for c in list(vs.unset()):
            try:
                cbar = constraint_complement(c)
            except ValueError:
                cbar = ~c
            m = cp.Model(cl + [cbar] + [v >= v.lb for v in Y])
            try:
                s = cp.SolverLookup.get("ortools", m)
                flag = s.solve(time_limit=self.time_limit)
                unsat = (not flag) and getattr(s, "cpm_status", None) is not None \
                    and s.cpm_status.exitstatus == ExitStatus.UNSATISFIABLE
            except Exception:
                flag, unsat = False, False

            if flag and all(v.value() is not None for v in Y) and self._is_irredundant(vs, Y):
                return list(Y)
            if unsat:
                vs.force_true(c)
                vs.propagate()
                cl = [c2 for c2 in vs.forced_true if c2 in vs.bias]

        return None

    def _is_irredundant(self, vs, Y):
        kappa = set(vs.kappa_under_T(Y))
        if not kappa:
            return False
        for clause in vs.clauses:
            reduced = vs._reduce_clause(clause)
            if reduced and reduced.issubset(kappa):
                return False
        return True
