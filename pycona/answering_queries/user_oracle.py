from .oracle import Oracle


class UserOracle(Oracle):
    """
    The Oracle is a human user, who directly answers the given queries
    """

    def __init__(self):
        """
        Initialize the UserOracle instance.
        """
        super().__init__()

    def answer_membership_query(self, Y=None):
        """
        Ask the user if the given values satisfy the constraints.

        :param Y: The values to be checked against the constraints.
        :return: A boolean indicating if the values satisfy the constraints.
        """
        # Prompt user for input
        response = input().strip().lower()
        while response not in ['yes', 'y', 'no', 'n']:
            response = input("Please answer with yes, y, no, or n").strip().lower()
        return response == 'yes' or response == 'y'

    def answer_recommendation_query(self, c=None):
        """
        Ask the user if the recommended constraint is part of the constraints.

        :param c: The recommended constraint to be checked.
        :return: A boolean indicating if the recommended constraint is part of the constraints.
        """
        # Prompt user for input
        response = input().strip().lower()
        while response not in ['yes', 'y', 'no', 'n']:
            response = input("Please answer with yes, y, no, or n").strip().lower()
        return response == 'yes' or response == 'y'

    def answer_generalization_query(self, C=None):
        """
        Ask the user if the given generalization of constraints is correct.

        :param C: The generalization of constraints to be checked.
        :return: A boolean indicating if the generalization of constraints is correct.
        """
        # Prompt user for input
        response = input().strip().lower()
        while response not in ['yes', 'y', 'no', 'n']:
            response = input("Please answer with yes, y, no, or n").strip().lower()
        return response == 'yes' or response == 'y'
