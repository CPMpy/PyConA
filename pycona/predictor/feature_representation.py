import re
from abc import ABC, abstractmethod
import numpy as np
from cpmpy.expressions.utils import is_any_list

from ..problem_instance import ProblemInstance
from ..utils import get_var_name, get_var_dims, get_scope, get_var_ndims, get_relation, get_constant
from .utils import get_divisors, average_difference

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
        self._max_arity = max(len(get_scope(rel)) for rel in self._lang)
        constants = [get_constant(rel) for rel in self._lang]
        self._max_constants = max(len(cnst) for cnst in constants)

        self._features['Var_name_same'] = 'Bool'

        for i in range(self._max_ndims):
            for j in range(self._max_arity):
                self._features[f"Dim{i}_var{j}_index"] = 'Int'
            self._features[f"Dim{i}_same"] = 'Bool'
            self._features[f"Dim{i}_max"] = 'Int'
            self._features[f"Dim{i}_min"] = 'Int'
            self._features[f"Dim{i}_avg"] = 'Real'
            self._features[f"Dim{i}_diff"] = 'Real'

        self._features[f"Relation"] = self._lang
        self._features[f"Arity"] = 'Int'
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
                for k in range(self._max_arity):
                    features.append(dim[j][k])
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


class FeaturesRelDimBlock(FeaturesRelDim):
    def __init__(self):
        """
        Initialize the FeaturesRelDimBlock with an optional problem instance.
        Feature representation that can be used with any classifier, from:
        Dimos Tsouros, Senne Berden, and Tias Guns.
        "Generalizing Constraint Models in Constraint Acquisition." AAAI, 2025

        :param instance: An instance of ProblemInstance, default is None.
        """
        super().__init__()

    def _init_features(self):
        """
        Initialize the features dictionary based on the problem instance.
        """
        variables = list(self.instance.variables.flatten())
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
        
        # Find max dimension length across all variable types
        max_dim_lengths = [
            max(dim_len[j] for dim_len in self.dim_lengths)
            for j in range(max(len(dim_len) for dim_len in self.dim_lengths))
        ]

        self._dim_divisors = []

        for i in range(len(self.var_names)):
            self._dim_divisors.append([])
            # Calculate divisors once for each max dimension
            for max_dim_len in max_dim_lengths:
                divisors = get_divisors(max_dim_len)
                self._dim_divisors[i].append(divisors)
        
        self._max_divisors = [max(len(div_list[i]) for div_list in self._dim_divisors) for i in range(len(self._dim_divisors[0]))]

        self._lang = self.instance.language
        constants = [get_constant(rel) for rel in self._lang]
        self._max_constants = max(len(cnst) for cnst in constants)

        # Replace the single relation feature with one-hot encoded features
        # Remove the original relation feature
        # self._features[f"Relation"] = {'Type': self._lang, 'Relation': 'lang'}
        
        # Add one feature per relation in the language
        for rel_idx, relation in enumerate(self._lang):
            self._features[f"Relation_{rel_idx}"] = {'Type': 'Bool', 'Relation': 'lang', 'RelationIndex': rel_idx}

        self._features[f"Arity"] = {'Type': 'Num', 'Relation': 'arity'}
        if self._max_constants > 0:
            self._features[f"Has Constant"] = {'Type': 'Bool', 'Relation': 'has_constant'}
            for cnst in range(self._max_constants):
                self._features[f"Constant parameter {cnst}"] = {'Type': 'Num', 'Relation': 'constant', 'Constant': cnst}

        self._features['Var_name_same'] = {'Type': 'Bool', 'Partition': 'name', 'Condition': 'same'}
        
        # Add dimension boolean features
        for i in range(self._max_ndims):
            self._features[f"Dim{i}_same"] = {'Type': 'Bool', 'Partition': 'dim', 'Condition': 'same', 'dim': i}
            # Add block boolean features for this dimension
            block = 0
            for di in range(self._max_divisors[i]):
                block += 1             
                self._features[f"Dim{i}_block{block}_same"] = {'Type': 'Bool', 'Partition': 'block', 'Condition': 'same', 'dim': i, 'divisor': self._dim_divisors[0][i][di]}

        # Then add all numerical features
        for i in range(self._max_ndims):
            self._features[f"Dim{i}_max"] = {'Type': 'Num', 'Sequence': 'dim', 'Condition': 'max', 'dim': i}
            self._features[f"Dim{i}_min"] = {'Type': 'Num', 'Sequence': 'dim', 'Condition': 'min', 'dim': i}
            self._features[f"Dim{i}_avg"] = {'Type': 'Num', 'Sequence': 'dim', 'Condition': 'avg', 'dim': i}
            self._features[f"Dim{i}_diff"] = {'Type': 'Num', 'Sequence': 'dim', 'Condition': 'diff', 'dim': i}
            
            # Add block numerical features for this dimension
            block = 0
            for di in range(self._max_divisors[i]):
                block += 1
                self._features[f"Dim{i}_block{block}_max"] = {'Type': 'Num', 'Sequence': 'block', 'Condition': 'max', 'dim': i, 'divisor': self._dim_divisors[0][i][di]}
                self._features[f"Dim{i}_block{block}_min"] = {'Type': 'Num', 'Sequence': 'block', 'Condition': 'min', 'dim': i, 'divisor': self._dim_divisors[0][i][di]}
                self._features[f"Dim{i}_block{block}_avg"] = {'Type': 'Num', 'Sequence': 'block', 'Condition': 'avg', 'dim': i, 'divisor': self._dim_divisors[0][i][di]}
                self._features[f"Dim{i}_block{block}_diff"] = {'Type': 'Num', 'Sequence': 'block', 'Condition': 'diff', 'dim': i, 'divisor': self._dim_divisors[0][i][di]}

        

    def featurize_constraint(self, c):
        """
        Returns the features associated with constraint c.

        :param c: The constraint to be featurized.
        :return: A list of features for the given constraint.
        """

        features = []
        scope = get_scope(c)

        # First add relation features (one-hot encoded)
        rel_idx = get_relation(c, self._lang)
        one_hot_relation = [i == rel_idx for i in range(len(self._lang))]
        features.extend(one_hot_relation)

        # Add arity and constant features
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

        # Add all boolean features first
        var_name = get_var_name(scope[0])
        var_name_same = all([(get_var_name(var) == var_name) for var in scope])
        features.append(var_name_same)

        vars_dims = [get_var_dims(var) for var in scope]
        dim = []
        
        # Add dimension and block boolean features
        for j in range(self._max_ndims):
            dim.append([vars_dims[i][j] for i in range(len(vars_dims)) if len(vars_dims[i]) > j])
            dimj_has = len(dim[j]) > 0
            
            # Add dimension same feature
            if dimj_has:
                dimj_same = all([dim_temp == dim[j][0] for dim_temp in dim[j]])
                features.append(dimj_same)
            else:
                features.append(True)

            # Add block same features
            for var_name in range(len(self.var_names)):
                vars_block = [[dim[j][var] // divisor for var in range(len(scope))
                           if self.var_names[var_name] == get_var_name(scope[var])]
                          for divisor in self._dim_divisors[var_name][j]]

                for l in range(len(self._dim_divisors[var_name][j])):
                    block_same = all([vars_block[l][var] == vars_block[l][0] for var in range(len(vars_block[l]))])
                    features.append(block_same)

        # Then add all numerical features
        for j in range(self._max_ndims):
            dimj_has = len(dim[j]) > 0
            
            # Add dimension numerical features
            if dimj_has:
                dimj_max = max(dim[j])
                features.append(dimj_max)
                dimj_min = min(dim[j])
                features.append(dimj_min)
                dimj_avg = np.mean(dim[j])
                features.append(dimj_avg)
                dimj_diff = average_difference(dim[j])
                features.append(dimj_diff)
            else:
                for _ in range(4):  # max, min, avg, diff
                    features.append(0)

            # divisors features --- per var_name, per dimension (up to max dimensions), per divisor (up to max divisors)
            # block same, max, min, average, spread
            # for each variable name (type of variable)
            for var_name in range(len(self.var_names)):

                vars_block = [[dim[j][var] // divisor for var in range(len(scope))
                               if self.var_names[var_name] == get_var_name(scope[var])]
                              for divisor in self._dim_divisors[var_name][j]]

                for l in range(len(self._dim_divisors[var_name][j])):
                    # Get max block value
                    block_max = max(vars_block[l])
                    features.append(block_max)
                    # Get min block value
                    block_min = min(vars_block[l])
                    features.append(block_min)
                    # Get average block value
                    block_avg = np.mean(vars_block[l])
                    features.append(block_avg)
                    # Calculate and append average difference between consecutive blocks
                    block_diff = average_difference(vars_block[l])
                    features.append(block_diff)

        return features
