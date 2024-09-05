from abc import ABC, abstractmethod

from .. import Metrics
from ..problem_instance import ProblemInstance


class CAEnv(ABC):
    """
    Abstract class interface for CA environments.
    """

    def __init__(self):
        """
        Initialize the CA system.
        """
        self._instance = None
        self.metrics = None
        self.verbose = 0
        self.converged = False

    def init_state(self, **kwargs):
        """ Initialize the state of the CA system. """
        self._converged = False

    @property
    def instance(self):
        """ Getter method for _instance """
        return self._instance

    @instance.setter
    def instance(self, instance):
        """ Setter method for _instance """
        self._instance = instance

    @property
    def metrics(self):
        """ Getter method for _metrics """
        return self._metrics

    @metrics.setter
    def metrics(self, metrics):
        """ Setter method for _metrics """
        self._metrics = metrics

    @property
    def verbose(self):
        """ Get the verbosity of the system """
        return self._verbose

    @verbose.setter
    def verbose(self, verbose):
        """ Set the verbosity of the system """
        self._verbose = verbose

    @property
    def converged(self):
        """ Get the convergence value """
        return self._converged

    @converged.setter
    def converged(self, converged):
        """ Set the convergence value """
        self._converged = converged
