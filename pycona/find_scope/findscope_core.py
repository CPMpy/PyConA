from abc import ABC, abstractmethod

from ..ca_environment.active_ca import ActiveCAEnv
from .findscope_obj import split_half


class FindScopeBase(ABC):
    """
    Abstract class interface for FindScope implementations
    """

    def __init__(self, ca_system: ActiveCAEnv = None, time_limit=0.2, *, split_func=split_half):
        """
        Initialize the FindScopeBase class.

        :param ca_system: The constraint acquisition system.
        :param time_limit: The time limit for findscope variable spliting.
        :param split_func: The function used to split the variables in findscope.
        """
        self.ca = ca_system
        self._time_limit = time_limit
        self.split_func = split_func

    @abstractmethod
    def run(self, scope):
        """
        Method that all FindScope implementations must implement to initialize its running state and run.

        :param scope: The scope to be processed.
        :raises NotImplementedError: If the method is not implemented.
        """
        assert self.ca is not None
        raise NotImplementedError

    @abstractmethod
    def _find_scope(self, *args, **kwargs):
        """
        Method that all FindScope implementations must implement: Finding the scope procedure.

        :raises NotImplementedError: If the method is not implemented.
        """
        raise NotImplementedError

    @property
    def ca(self):
        """
        Get the constraint acquisition system.

        :return: The constraint acquisition system.
        """
        return self._ca

    @ca.setter
    def ca(self, ca_system: ActiveCAEnv):
        """
        Set the constraint acquisition system.

        :param ca_system: The constraint acquisition system.
        """
        if ca_system is not None:
            self._ca = ca_system
            if self._ca.find_scope != self:
                self._ca.find_scope = self

    @property
    def time_limit(self):
        """
        Get the time limit for findscope variable spliting.

        :return: The time limit.
        """
        return self._time_limit

    @time_limit.setter
    def time_limit(self, time_limit):
        """
        Set the time limit for findscope variable spliting.

        :param time_limit: The time limit.
        """
        self._time_limit = time_limit

    @property
    def split_func(self):
        """
        Get the split function to be used in findscope.

        :return: The split function.
        """
        return self._split_func

    @split_func.setter
    def split_func(self, split_func):
        """
        Set the split function to be used in findscope.

        :param split_func: The split function.
        :raises AssertionError: If the split function is not available.
        """
        from .. import Objectives
        assert split_func in Objectives.findscope_objectives(), "Split function given for FindScope is not available"
        self._split_func = split_func
