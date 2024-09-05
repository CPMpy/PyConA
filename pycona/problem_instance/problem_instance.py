import copy
import warnings

import cpmpy as cp
from cpmpy.expressions.core import Expression
from cpmpy.expressions.variables import _NumVarImpl, NDVarArray

from cpmpy.expressions.utils import is_any_list
from itertools import combinations
from ..utils import get_scope, replace_variables


class ProblemInstance:
    """ Class representing a problem instance with constraints, variables, and parameters. """

    def __init__(self, *, init_cl=None, variables=None, params=None, language=None, name=None, bias=None, excluded=None,
                 visualize=None):
        """
        Initialize the ProblemInstance with optional constraints, variables, parameters, and name.

        :param init_cl: The set of initially known constraints of the problem, default is an empty list.
        :param language: A list of relations to be used as the language, default is an empty list.
        :param bias: A list of candidate constraints, default is an empty list.
        :param variables: A list of Variable objects, default is an empty list.
        :param params: A dictionary of parameters, default is an empty dictionary.
        :param name: The name of the problem instance, default is an empty string.
        :param excluded: A list of excluded constraints, default is an empty list.
        :param visualize: A function to visualize assignments of the problem instance, default is a simple print.
        """
        self._X = []

        self.variables = variables
        self._params = params if params is not None else dict()
        self._name = name if name is not None else ""

        self._cl = init_cl if init_cl is not None else []
        self.language = language if language is not None else []
        self._bias = bias if bias is not None else []
        self._excluded_constraints = excluded if excluded is not None else []

        self._visualize = visualize if visualize is not None else print

        if not is_any_list(self._cl) or \
           not all(isinstance(c, Expression) for c in self._cl):
            raise TypeError(f"'constraints' argument in ProblemInstance should be a list of Constraint objects: "
                            f"{self._cl}")
        if not isinstance(self._variables, NDVarArray) and \
           not (is_any_list(self._variables) and all(isinstance(v, _NumVarImpl) for v in list(self._variables))):
            raise TypeError(f"'variables' argument in ProblemInstance should be a list of Variable objects: "
                            f"{type(self._variables)}")
        if not isinstance(self._params, dict):
            raise TypeError("'params' argument in ProblemInstance should be a dictionary of parameters")

    @property
    def cl(self):
        """
        Get the list of constraints.

        :return: The list of constraints.
        """
        return self._cl

    @cl.setter
    def cl(self, cl):
        """
        Set the list of constraints.

        :param cl: The new list of constraints.
        """
        self._cl = cl

    @property
    def bias(self):
        """
        Get the list of candidate constraints.

        :return: The list of candidate constraints.
        """
        return self._bias

    @bias.setter
    def bias(self, B):
        """
        Set the list of candidate constraints.

        :param B: The new list of candidate constraints.
        """
        self._bias = B

    @property
    def excluded_cons(self):
        """
        Get the list of excluded constraints.

        :return: The list of excluded constraints.
        """
        return self._excluded_constraints

    @excluded_cons.setter
    def excluded_cons(self, C):
        """
        Set the list of excluded constraints.

        :param C: The new list of excluded constraints.
        """
        self._excluded_constraints = C

    @property
    def variables(self):
        """
        Get the list of variables.

        :return: The variables.
        """
        return self._variables

    @variables.setter
    def variables(self, vars):
        """
        Set the list of variables.

        :param vars: The new variables.
        """
        self._variables = vars
        if vars is not None:
            self.X = list(self._variables.flatten())

    @property
    def X(self):
        """
        Get the list of flattened variables.

        :return: The list of flattened variables.
        """
        return self._X

    @X.setter
    def X(self, X):
        """
        Set the list of variables to operate on.

        :param X: The new list of flattened variables.
        """
        self._X = X

    @property
    def language(self):
        """
        Get the language.

        :return: The list language.
        """
        return self._language

    @language.setter
    def language(self, lang):
        """
        Set the language. Must be a list of cpmpy expressions

        :param lang: The new language.
        """
        assert all(isinstance(r, Expression) for r in lang)
        self._language = lang

    @property
    def params(self):
        """
        Get the dictionary of parameters.

        :return: The dictionary of parameters.
        """
        return self._params

    @params.setter
    def params(self, params):
        """
        Set the dictionary of parameters.

        :param params: The new dictionary of parameters.
        """
        self._params = params

    @property
    def name(self):
        """
        Get the name of the problem instance.

        :return: The name of the problem instance.
        """
        return self._name

    @name.setter
    def name(self, name):
        """
        Set the name of the problem instance.

        :param name: The new name of the problem instance.
        """
        self._name = name

    @property
    def visualize(self):
        """
        Get the visualize function of the problem instance.
        """
        return self._visualize

    @visualize.setter
    def visualize(self, visualize):
        """
        Set the visualize function of the problem instance.
        """
        self._visualize = visualize

    def get_cpmpy_model(self):
        if len(self._cl) == 0:
            warnings.warn("The model is empty, as no constraint is learned yet for this instance.")
        return cp.Model(self._cl)

    def construct_bias(self):
        """
        Construct the bias (candidate constraints) for the problem instance.
        """

        all_cons = []

        X = list(self.X)

        for relation in self.language:

            abs_vars = get_scope(relation)

            combs = combinations(X, len(abs_vars))

            for comb in combs:
                replace_dict = dict()
                for i, v in enumerate(comb):
                    replace_dict[abs_vars[i]] = v
                constraint = replace_variables(relation, replace_dict)
                all_cons.append(constraint)

        self.bias = all_cons

    def construct_bias_for_var(self, v1):
        """
        Construct the bias (candidate constraints) for a specific variable.

        :param v1: The variable for which to construct the bias.
        """
        
        all_cons = []
        X = list(set(self.X) - {v1})

        for relation in self.language:
            abs_vars = get_scope(relation)

            combs = combinations(X, len(abs_vars)-1)

            for comb in combs:
                replace_dict = {abs_vars[0]: v1}
                for i, v in enumerate(comb):
                    replace_dict[abs_vars[i+1]] = v
                constraint = replace_variables(relation, replace_dict)
                all_cons.append(constraint)

        self.bias = all_cons

    def __str__(self):
        """
        Return a string representation of the ProblemInstance.

        :return: A string representation of the ProblemInstance.
        """
        parts = [f"ProblemInstance: "]
        if self._name is not None and len(self._name) > 0:
            parts.append(f"\nName {self._name}.")
        if self._params is not None and len(self._params) > 0:
            parts.append(f"\nParameters {self._params}.")
        if self._variables is not None:
            parts.append(f"\nVariables: {self._variables}.")
        if self._cl is not None and len(self._cl) > 0:
            parts.append(f"\nConstraints: {self._cl}.")
        if self.language is not None:
            parts.append(f"\nLanguage: {self.language}.")
        return "\n".join(parts)

    def copy(self):
        """
        Create a copy of the ProblemInstance.

        :return: A copy of the ProblemInstance.
        """
        instance = copy.copy(self)

        instance._X = self.X.copy()
        instance.cl = self.cl.copy()
        instance._language = self._language.copy()
        instance.bias = self.bias.copy()
        instance._excluded_constraints = self._excluded_constraints.copy()

        return instance
