import cpmpy as cp
from cpmpy.transformations.normalize import toplevel_list
from ..answering_queries.constraint_oracle import ConstraintOracle
from ..problem_instance import ProblemInstance, absvar


def construct_purdey_problem():
    """
    :Description: A small Purdey-style logic grid puzzle, with three entities and three categories (drink, store, pet).
    The clues with a constant (e.g., milk is drunk by the second person) are given as known constraints, the rest of
    the clues (equalities, inequalities and adjacency) have to be learned.
    :return: a ProblemInstance object, along with a constraint-based oracle
    """
    parameters = {"entities": 3}

    # Implicit base category is person index 1..3.
    # Rows encode where each item of a category is assigned.
    grid = cp.intvar(1, 3, shape=(3, 3), name="purdey")
    drink, store, pet = grid

    tea, coffee, milk = range(3)
    bakery, butcher, grocer = range(3)
    cat, dog, bird = range(3)

    model = cp.Model()

    for row in grid:
        model += cp.AllDifferent(row).decompose()

    # Clues
    model += drink[tea] == store[bakery]
    model += drink[coffee] != store[butcher]
    model += pet[dog] == store[grocer]
    model += abs(pet[bird] - drink[coffee]) == 1

    if not model.solve():
        raise Exception("The problem has no solution")

    C_T = list(set(toplevel_list(model.constraints)))

    # Create the language:
    AV = absvar(2)  # create abstract vars - as many as maximum arity

    # create abstract relations using the abstract vars
    lang = [
        AV[0] == AV[1],
        AV[0] != AV[1],
        AV[0] < AV[1],
        AV[0] > AV[1],
        AV[0] >= AV[1],
        AV[0] <= AV[1],
        abs(AV[0] - AV[1]) == 1,
    ]

    instance = ProblemInstance(
        variables=grid,
        init_cl=[drink[milk] == 2, pet[cat] != 2, store[butcher] != 1],
        params=parameters,
        language=lang,
        name="purdey",
    )

    oracle = ConstraintOracle(C_T)

    return instance, oracle
