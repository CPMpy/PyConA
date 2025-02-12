import cpmpy as cp
from cpmpy.transformations.normalize import toplevel_list
from ..answering_queries.constraint_oracle import ConstraintOracle
from ..problem_instance import ProblemInstance, absvar


def construct_nurse_rostering(shifts_per_day=3, num_days=5, num_nurses=8, nurses_per_shift=2):
    """
    :return: a ProblemInstance object, along with a constraint-based oracle
    """

    # Create a dictionary with the parameters
    parameters = {"shifts_per_day": shifts_per_day, "num_days": num_days, "num_nurses": num_nurses,
                  "nurses_per_shift": nurses_per_shift}

    # Define the variables
    roster_matrix = cp.intvar(1, num_nurses, shape=(num_days, shifts_per_day, nurses_per_shift), name="shifts")

    # Define the constraints
    model = cp.Model()

    # Constraint: Each shift in a day must be assigned to a different nurse
    for day in range(num_days):
        model += cp.AllDifferent(roster_matrix[day, ...]).decompose()

    # Constraint: The last shift of a day cannot have the same nurse as the first shift of the next day
    for day in range(num_days - 1):
        model += cp.AllDifferent(roster_matrix[day, shifts_per_day - 1], roster_matrix[day + 1, 0]).decompose()

    if not model.solve():
        raise Exception("The problem has no solution")

    C_T = list(set(toplevel_list(model.constraints)))

    # Create the language:
    AV = absvar(2)  # create abstract vars - as many as maximum arity

    # create abstract relations using the abstract vars
    lang = [AV[0] == AV[1], AV[0] != AV[1], AV[0] < AV[1], AV[0] > AV[1], AV[0] >= AV[1], AV[0] <= AV[1]]

    instance = ProblemInstance(variables=roster_matrix, params=parameters, language=lang,
                               name=f"nurse_rostering_nurses{num_nurses}_shifts{shifts_per_day}_days{num_days}_"
                                    f"nurses_per_shift{nurses_per_shift}")

    oracle = ConstraintOracle(C_T)

    return instance, oracle

