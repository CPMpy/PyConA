import cpmpy as cp
from cpmpy.transformations.normalize import toplevel_list
from ..answering_queries.constraint_oracle import ConstraintOracle
from ..problem_instance import ProblemInstance, absvar


def construct_zebra_problem():
    """
    :Description: The zebra puzzle is a well-known logic puzzle. Five houses, each of a different color, are occupied by men of
    different nationalities, with different pets, drinks and cigarettes. The puzzle is to find out who owns the zebra.
    The clues that fix an item to a house (milk in the middle house, Norwegian in the first house) are given as
    known constraints, the rest of the clues have to be learned.
    :return: a ProblemInstance object, along with a constraint-based oracle
    """
    parameters = {"houses": 5}

    # Each row encodes a category; each value stores the house index (1..5)
    # for that category item.
    grid = cp.intvar(1, 5, shape=(5, 5), name="zebra")
    color, nation, drink, smoke, pet = grid

    # Item indices per category
    red, green, ivory, yellow, blue = range(5)
    englishman, spaniard, ukrainian, norwegian, japanese = range(5)
    coffee, tea, milk, orange_juice, water = range(5)
    old_gold, kools, chesterfields, lucky_strike, parliaments = range(5)
    dog, snails, fox, horse, zebra = range(5)

    model = cp.Model()

    # One house per item in each category
    for row in grid:
        model += cp.AllDifferent(row).decompose()

    # Zebra puzzle clues
    model += nation[englishman] == color[red]
    model += nation[spaniard] == pet[dog]
    model += drink[coffee] == color[green]
    model += nation[ukrainian] == drink[tea]
    model += smoke[old_gold] == pet[snails]
    model += smoke[kools] == color[yellow]
    model += abs(smoke[chesterfields] - pet[fox]) == 1
    model += abs(smoke[kools] - pet[horse]) == 1
    model += smoke[lucky_strike] == drink[orange_juice]
    model += nation[japanese] == smoke[parliaments]
    model += abs(nation[norwegian] - color[blue]) == 1

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
        AV[0] + 1 == AV[1],
        AV[1] + 1 == AV[0],
        abs(AV[0] - AV[1]) == 1,
    ]

    instance = ProblemInstance(
        variables=grid,
        init_cl=[drink[milk] == 3, nation[norwegian] == 1],
        params=parameters,
        language=lang,
        name="zebra",
    )

    oracle = ConstraintOracle(C_T)

    return instance, oracle
