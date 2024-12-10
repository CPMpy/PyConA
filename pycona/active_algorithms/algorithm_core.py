from abc import ABC, abstractmethod

from .. import Metrics
from ..problem_instance import ProblemInstance
from ..answering_queries import Oracle, UserOracle
from ..ca_environment.active_ca import ActiveCAEnv

class AlgorithmCAInteractive(ABC):
    """
    Abstract base class for ICA (Interactive Constraint Acquisition) active_algorithms.
    """

    def __init__(self, ca_env: ActiveCAEnv = None):
        """
        Initialize the AlgorithmCAInteractive with a constraint acquisition environment.

        :param ca_env: A ca environment.
        """
        self.env = ca_env if ca_env is not None else ActiveCAEnv()

    @abstractmethod
    def learn(self, instance: ProblemInstance, oracle: Oracle = UserOracle(), verbose=0, metrics: Metrics = None):
        """
        Abstract method to learn constraints. Must be implemented by subclasses.

        :param instance: the problem instance to acquire the constraints for
        :param oracle: An instance of Oracle, default is to use the user as the oracle.
        :param verbose: Verbosity level, default is 0.
        :param metrics: statistics logger during learning
        :return: the learned instance
        """
        raise NotImplementedError

    @property
    def env(self):
        """
        Get the constraint acquisition environment.

        :return: The constraint acquisition environment.
        """
        return self._env

    @env.setter
    def env(self, env: ActiveCAEnv):
        """
        Set the constraint acquisition environment and assign this algorithm to it.

        :param env: The constraint acquisition environment.
        """
        self._env = env

    def __repr__(self):
        return f"{self.__class__.__name__}"
