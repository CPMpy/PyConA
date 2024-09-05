import numpy as np
from cpmpy.expressions.core import Expression

from .ca_env_core import CAEnv
from ..metrics import Metrics
from ..utils import get_kappa


class ActiveCAEnv(CAEnv):
    """
    Class interface for the interactive CA systems. Using all CA components (Algorithm, Query generation, FindScope,
    FindC etc.), storing the necessary elements and providing functionality to update the state of the system as needed.
    """

    def __init__(self, qgen: 'QGenBase' = None,
                 find_scope: 'FindScopeBase' = None, findc: 'FindCBase' = None):
        """
        Initialize with an optional problem instance, oracle, and metrics.

        :param qgen: An instance of QGenBase, default is None.
        :param find_scope: An instance of FindScopeBase, default is None.
        :param findc: An instance of FindCBase, default is None.
        """
        super().__init__()

        self._oracle = None

        from ..query_generation import PQGen
        from ..query_generation.qgen_obj import obj_max_viol
        self.qgen = qgen if qgen is not None else PQGen(objective_function=obj_max_viol)

        from ..find_scope import FindScope2
        self.find_scope = find_scope if find_scope is not None else FindScope2()

        from ..find_constraint import FindC
        self.findc = findc if findc is not None else FindC()
        self._last_answer = True

    def init_state(self, instance, oracle, verbose, metrics=None):
        """ Initialize the state of the CA system. """
        super().init_state()

        self.instance = instance.copy()
        self.oracle = oracle
        self.verbose = verbose
        self.metrics = metrics if metrics is not None else Metrics()

        self.qgen.env = self
        self.find_scope.ca = self
        self.findc.ca = self

    def run_query_generation(self):
        """ Run the query generation process. """
        Y = self.qgen.generate()
        return Y

    def run_find_scope(self, Y):
        """ Run the find scope process. """
        scope = self.find_scope.run(Y)
        return scope

    def run_findc(self, scope):
        """ Run the find constraint process. """
        c = self.findc.run(scope)
        return c

    @property
    def qgen(self):
        """ Getter method for _qgen """
        return self._qgen

    @qgen.setter
    def qgen(self, qgen: 'QGenBase'):
        """ Setter method for _qgen """
        from ..query_generation import QGenBase
        assert isinstance(qgen, QGenBase)
        self._qgen = qgen
        self._qgen.env = self

    @property
    def find_scope(self):
        """ Getter method for _find_scope """
        return self._find_scope

    @find_scope.setter
    def find_scope(self, find_scope: 'FindScopeBase'):
        """ Setter method for _find_scope """
        from ..find_scope import FindScopeBase, FindScope2
        assert isinstance(find_scope, FindScopeBase)
        self._find_scope = find_scope
        self._find_scope.ca = self

    @property
    def findc(self):
        """ Getter method for _findc """
        return self._findc

    @findc.setter
    def findc(self, findc: 'FindCBase'):
        """ Setter method for _findc """
        from ..find_constraint import FindCBase
        assert isinstance(findc, FindCBase)
        self._findc = findc
        self._findc.ca = self

    @property
    def oracle(self):
        """ Getter method for _oracle """
        return self._oracle

    @oracle.setter
    def oracle(self, oracle):
        """ Setter method for _oracle """
        self._oracle = oracle

    @property
    def last_answer(self):
        """ Get the last answer (bool) """
        return self._last_answer

    @last_answer.setter
    def last_answer(self, last_answer):
        """ Set the last answer (bool) """
        self._last_answer = last_answer

    def ask_membership_query(self, Y=None):
        """
        Ask a membership query to the oracle.

        :param Y: Optional. A subset of variables to be used in the query. If None, all variables are used.
        :return: The oracle's answer to the membership query (True/False).
        """
        X = self.instance.X
        if Y is None:
            Y = X
        e = self.instance.variables.value()
        value = np.zeros(e.shape, dtype=int)

        # Create a truth table numpy array
        sel = np.array([item in set(Y) for item in list(self.instance.variables.flatten())]).reshape(self.instance.variables.shape)

        # Variables present in the partial query
        value[sel] = e[sel]

        # Post the query to the user/oracle
        if self.verbose >= 3:
            print(f"Query{self.metrics.membership_queries_count}: is this a solution?")
            self.instance.visualize(value)
        if self.verbose >= 4:
            print("violated from B: ", get_kappa(self.instance.bias, Y))

        # Oracle answers
        self.last_answer = self.oracle.answer_membership_query(Y)
        if self.verbose >= 3:
            print("Answer: ", ("Yes" if self.last_answer else "No"))

        # For the evaluation metrics
        if self.metrics:
            self.metrics.increase_membership_queries_count()
            self.metrics.increase_queries_size(len(Y))
            self.metrics.asked_query()
        if self.verbose == 1:
            print(".", end="")

        return self.last_answer

    def ask_recommendation_query(self, c):
        """
        Ask a recommendation query to the oracle.

        :param c: The constraint to be recommended.
        :return: The oracle's answer to the recommendation query (True/False).
        """
        assert isinstance(c, Expression), "Recommendation queries need constraints as input"
        if self.verbose >= 3:
            print(f"Rec query: is this a constraint of the problem? {c}")

        answer = self.oracle.answer_recommendation_query(c)
        if self.verbose >= 3:
            print("Answer: ", ("Yes" if answer else "No"))

        # For the evaluation metrics
        if self.metrics:
            self.metrics.increase_recommendation_queries_count()
            self.metrics.asked_query()

        return answer

    def ask_generalization_query(self, c, C):
        """
        Ask a generalization query to the oracle.

        :param c: The constraint to be generalized.
        :param C: A list of constraints to which the generalization is applied.
        :return: The oracle's answer to the generalization query (True/False).
        """
        assert isinstance(c, Expression), "Generalization queries first input needs to be a constraint"
        assert isinstance(C, list), "Generalization queries second input needs to be a list of constraints"
        assert all(isinstance(c1, Expression) for c1 in C), "Generalization queries second input needs to be " \
                                                           "a list of constraints"
        if self.verbose >= 3:
            print(f"Generalization query: Can we generalize constraint {c} to all in {C}?")

        answer = self.oracle.answer_generalization_query(C)
        if self.verbose >= 3:
            print("Answer: ", ("Yes" if answer else "No"))

        # For the evaluation metrics
        if self.metrics:
            self.metrics.increase_generalization_queries_count()
            self.metrics.asked_query()

        return answer

    def remove_from_bias(self, C):
        """
        Remove given constraints from the bias (candidates)

        :param C: list of constraints to be removed from B
        """
        if isinstance(C, Expression):
            C = [C]
        assert isinstance(C, list), "remove_from_bias accepts as input a list of constraints or a constraint"

        if self.verbose >= 3:
            print(f"removing the following constraints from bias: {C}")

        self.instance.bias = list(set(self.instance.bias) - set(C))

    def add_to_cl(self, C):
        """
        Add the given constraints to the list of learned constraints

        :param C: Constraints to add to CL
        """
        if isinstance(C, Expression):
            C = [C]
        assert isinstance(C, list), "add_to_cl accepts as input a list of constraints or a constraint"

        if self.verbose >= 3:
            print(f"adding the following constraints to C_L: {C}")

        # Add constraint(s) c to the learned network and remove them from the bias
        self.instance.cl.extend(C)
        self.instance.bias = list(set(self.instance.bias) - set(C))

        self.metrics.cl += 1
        if self.verbose == 1:
            print("L", end="")
