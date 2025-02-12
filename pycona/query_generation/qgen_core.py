from abc import ABC, abstractmethod
from ..ca_environment.active_ca import ActiveCAEnv


class QGenBase(ABC):
    """
    Abstract class interface for QGen implementations.
    """

    def __init__(self, ca_env: ActiveCAEnv = None, time_limit=2):
        """
        Initialize the QGenBase with the given CA environment and time limit.

        :param ca_env: The CA environment used.
        :param time_limit: Overall time limit.
        """
        self._env = ca_env
        self._time_limit = time_limit

    @abstractmethod
    def generate(self, Y=None):
        """
        Method that all QGen implementations must implement to generate a query.
        """
        raise NotImplementedError

    @property
    def env(self):
        """
        Get the CA environment.

        :return: The CA environment.
        """
        return self._env

    @env.setter
    def env(self, ca_env: ActiveCAEnv = None):
        """
        Set the CA environment.

        :param ca_env: The CA environment to set.
        """
        self._env = ca_env

    @property
    def time_limit(self):
        """
        Get the time limit.

        :return: The time limit.
        """
        return self._time_limit

    @time_limit.setter
    def time_limit(self, time_limit):
        """
        Set the time limit.

        :param time_limit: The time limit to set.
        """
        self._time_limit = time_limit
