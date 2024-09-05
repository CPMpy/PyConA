import numpy as np
from cpmpy.expressions.variables import NDVarArray, _genname, _NumVarImpl


def absvar(length: int = 1, name=None):
    """
    Abstract variables, to be used for the language of CA

    Arguments:
    shape -- the length of the vector of variables (int, default: 1)
    name -- name to give to the abstract variables (string, default: None)
    """
    if length == 0 or length is None:
        raise ValueError
    if length == 1:
        return _AbstracVar(name=name)

    # create base data
    data = np.array([_AbstracVar(name=_genname(name, idxs)) for idxs in np.ndindex(length)])  # repeat new instances
    # insert into custom ndarray
    return NDVarArray(length, dtype=object, buffer=data)


class _AbstracVar(_NumVarImpl):
    """
    Abstract variable, with only a name. Inherits from CPMpy (https://github.com/CPMpy/cpmpy) abstract variable class

    To be used to create the language for CA.
    """
    counter = 0

    def __init__(self, name):
        if name is None:
            name = "AV{}".format(_AbstracVar.counter)
            _AbstracVar.counter += 1  # static counter
        self.name = name

    def is_bool(self):
        """ is it a Boolean (return type) Operator?
        """
        return NotImplementedError("Abstract variable is not supposed to be used")

    def value(self):
        """ the value obtained in the last solve call
            (or 'None')
        """
        return NotImplementedError("Abstract variable is not supposed to be used")

    def get_bounds(self):
        """ the lower and upper bounds"""
        return NotImplementedError("Abstract variable is not supposed to be used")

    def clear(self):
        """ clear the value obtained from the last solve call
        """
        return NotImplementedError("Abstract variable is not supposed to be used")

    def __repr__(self):
        return self.name

    # for sets/dicts. Because names are unique, so is the str repr
    def __hash__(self):
        return hash(self.name)


# Creating basic languages
AV = absvar(4) # create abstract vars

langEqNeq = [AV[0] == AV[1], AV[0] != AV[1]]

langBasic = [AV[0] == AV[1], AV[0] != AV[1], AV[0] < AV[1], AV[0] > AV[1], AV[0] >= AV[1], AV[0] <= AV[1]]

langDist = [AV[0] == AV[1], AV[0] != AV[1], AV[0] < AV[1], AV[0] > AV[1], AV[0] >= AV[1], AV[0] <= AV[1],
            abs(AV[0] - AV[1]) != abs(AV[2] - AV[3]), abs(AV[0] - AV[1]) == abs(AV[2] - AV[3])]
