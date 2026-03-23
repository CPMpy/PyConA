import cpmpy as cp
from cpmpy.transformations.normalize import toplevel_list
from ..answering_queries.constraint_oracle import ConstraintOracle
from ..problem_instance import ProblemInstance, absvar


def construct_latin_squares(grid_size=4):
    """
    :return: a ProblemInstance object, along with a constraint-based oracle
    """

    # Create a dictionary with the parameters
    parameters = {"grid_size": grid_size}

    # Variables
    grid = cp.intvar(1, grid_size, shape=(grid_size, grid_size), name="grid")

    model = cp.Model()

    # Constraints on rows and columns
    for row in grid:
        model += cp.AllDifferent(row).decompose()

    for col in grid.T:  # numpy's Transpose
        model += cp.AllDifferent(col).decompose()

    C_T = list(set(toplevel_list(model.constraints)))

    # Create the language:
    AV = absvar(2)  # create abstract vars - as many as maximum arity

    # create abstract relations using the abstract vars
    lang = [AV[0] == AV[1], AV[0] != AV[1], AV[0] < AV[1], AV[0] > AV[1], AV[0] >= AV[1], AV[0] <= AV[1]]

    instance = ProblemInstance(variables=grid, params=parameters, language=lang, name=f"latin_squares_{grid_size}")

    oracle = ConstraintOracle(C_T)

    return instance, oracle
