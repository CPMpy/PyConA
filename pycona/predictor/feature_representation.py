import re
from abc import ABC, abstractmethod
import numpy as np
from cpmpy.expressions.utils import is_any_list

from ..problem_instance import ProblemInstance
from ..utils import get_var_name, get_var_dims, get_scope, get_var_ndims, get_relation, get_constant


class FeatureRepresentation(ABC):
    """
    Abstract base class for feature representation of a problem instance.
    """

    def __init__(self):
        """
        Initialize the FeatureRepresentation with an optional problem instance.

        """
        self._instance = None
        self._features = dict()  # dictionary with the names of the features and their respective type

    def _init_features(self):
        """
        Initialize the features' dictionary. Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def featurize_constraint(self, c):
        """
        Featurize a given constraint. Must be implemented by subclasses.

        :param c: The constraint to be featurized.
        """
        raise NotImplementedError

    def featurize_constraints(self, C):
        """
        Returns the features associated with constraints C.

        :param C: The constraints to be featurized.
        :return: A list with the feature values for the given constraints.
        """

        assert is_any_list(C), "Parameter C must be a list. Use 'featurize_constraint' for a single constraint"

        list_of_features = []
        for c in C:
            list_of_features.append(self.featurize_constraint(c))

        return list_of_features

    @property
    def instance(self):
        """
        Get the problem instance.
        """
        return self._instance

    @instance.setter
    def instance(self, instance):
        """
        Set the problem instance.
        """
        self._instance = instance
        self._init_features()

    @property
    def features(self):
        """
        Get the features' dictionary.
        """
        return self._features


class FeaturesRelDim(FeatureRepresentation):
    def __init__(self):
        """
        Initialize the FeaturesRelDim with an optional problem instance.
        Feature representation that can be used with any classifier, from:
        Dimos Tsouros, Senne Berden, and Tias Guns.
        "Learning to Learn in Interactive Constraint Acquisition." AAAI, 2024

        :param instance: An instance of ProblemInstance, default is None.
        """
        super().__init__()

    def _init_features(self):
        """
        Initialize the features dictionary based on the problem instance.
        """
        variables = self.instance.X
        self.var_names = list({get_var_name(x) for x in variables})
        self._max_ndims = max(get_var_ndims(x) for x in variables)

        var_dims = [
            [get_var_dims(x) for x in variables if get_var_name(x) == self.var_names[i]]
            for i in range(len(self.var_names))
        ]
        self.dim_lengths = [
            [
                np.max([var_dims[i][k][j] for k in range(len(var_dims[i]))]) + 1
                for j in range(len(var_dims[i][0]))
            ]
            for i in range(len(var_dims))
        ]
        self._lang = self.instance.language
        constants = [get_constant(rel) for rel in self._lang]
        self._max_constants = max(len(cnst) for cnst in constants)

        self._features['Var_name_same'] = 'Bool'

        for i in range(self._max_ndims):
            self._features[f"Dim{i}_same"] = 'Bool'
            self._features[f"Dim{i}_max"] = 'Int'
            self._features[f"Dim{i}_min"] = 'Int'
            self._features[f"Dim{i}_avg"] = 'Real'
            self._features[f"Dim{i}_diff"] = 'Real'

        self._features[f"Relation"] = self._lang
        self._features[f"Arity"] = 'Real'
        if self._max_constants > 0:
            self._features[f"Has Constant"] = 'Bool'
            for cnst in range(self._max_constants):
                self._features[f"Constant parameter {cnst}"] = 'Real'

    def featurize_constraint(self, c):
        """
        Returns the features associated with constraint c.

        :param c: The constraint to be featurized.
        :return: A list of features for the given constraint.
        """

        features = []
        scope = get_scope(c)
        var_name = get_var_name(scope[0])
        var_name_same = all([(get_var_name(var) == var_name) for var in scope])
        features.append(var_name_same)

        vars_dims = [get_var_dims(var) for var in scope]
        dim = []
        for j in range(self._max_ndims):
            dim.append([vars_dims[i][j] for i in range(len(vars_dims)) if len(vars_dims[i]) > j])
            dimj_has = len(dim[j]) > 0
            if dimj_has:
                dimj_same = all([dim_temp == dim[j][0] for dim_temp in dim[j]])
                features.append(dimj_same)
                dimj_max = max(dim[j])
                features.append(dimj_max)
                dimj_min = min(dim[j])
                features.append(dimj_min)
                dimj_avg = np.mean(dim[j])
                features.append(dimj_avg)
                dimj_diff = abs(np.mean(np.diff(dim[j])))
                features.append(dimj_diff)
            else:
                features.append(True)
                for _ in range(4):
                    features.append(0)
                features.append(0.0)

        con_in_gamma = get_relation(c, self._lang)

        features.append(con_in_gamma)
        arity = len(scope)
        features.append(arity)

        if self._max_constants > 0:
            num = get_constant(c)
            has_const = len(num) > 0
            features.append(has_const)
            for cnst in range(self._max_constants):
                try:
                    features.append(int(num[cnst]))
                except (IndexError, ValueError):
                    features.append(-1)

        return features


class FeaturesSimpleRel(FeatureRepresentation):
    def __init__(self):
        """
        Initialize the FeaturesRelDim with an optional problem instance.

        Simple feature representation storing only the constraint relation, from:
        Dimos Tsouros, Senne Berden, and Tias Guns. "Guided Bottom-Up Interactive Constraint Acquisition." CP, 2023
        """
        super().__init__()

    def _init_features(self):
        """
        Initialize the features dictionary based on the problem instance.
        """
        assert self.instance is not None
        self._lang = self.instance.language
        self._features[f"Relation"] = self._lang

    def featurize_constraint(self, c):
        """
        Returns the features associated with constraint c.

        :param c: The constraint to be featurized.
        :return: A list of features for the given constraint.
        """
        con_in_gamma = self._lang[get_relation(c, self._lang)]

        return [con_in_gamma]
