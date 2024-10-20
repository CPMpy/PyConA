from abc import ABC, abstractmethod


class Oracle(ABC):
    """
    Abstract base class representing an Oracle used in CA that can answer different types of queries

    This class provides an interface for the oracle. Implementing classes should define the behavior for
    answering the queries.
    """

    def __init__(self, verbose=0):
        """
        Initialize the Oracle instance.

        :param verbose: The verbosity level of the oracle.
        """
        self._verbose = verbose

    @abstractmethod
    def answer_membership_query(self, Y):
        """
        Answer a membership query.

        Determines whether the given assignment on Y is a solution or not.

        :param Y: The input values to be checked for membership.
        :return: A boolean indicating a positive or negative answer.
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def answer_recommendation_query(self, c):
        """
        Answers a recommendation query

        Determines if a recommended constraint is part of the problem or not.

        :param c: The recommended constraint.
        :return: A boolean indicating a positive or negative answer.
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def answer_generalization_query(self, C):
        """
        Answer a generalization query.

        Determines if the recommended generalization is correct or not.

        :param C: The constraints the generalization will generate.
        :return: A boolean indicating a positive or negative answer.
        """
        raise NotImplementedError("Subclasses must implement this method")

    @property
    def verbose(self):
        """
        Get the verbosity of the oracle

        :return: The verbosity level of the oracle.
        """
        return self._verbose

    @verbose.setter
    def verbose(self, verbose):
        """
        Set the verbosity of the oracle

        :param verbose: The verbosity level to be set.
        """
        self._verbose = verbose

    def __repr__(self):
        return f"{self.__class__.__name__}"
