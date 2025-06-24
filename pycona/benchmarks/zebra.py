import cpmpy as cp
from cpmpy.transformations.normalize import toplevel_list
from ..answering_queries.constraint_oracle import ConstraintOracle
from ..problem_instance import ProblemInstance, absvar


def construct_zebra_problem():
    """
    :Description: The zebra puzzle is a well-known logic puzzle. Five houses, each of a different color, are occupied by men of 
    different nationalities, with different pets, drinks and cigarettes. The puzzle is to find out who owns the zebra.
    The puzzle has 15 clues that help determine the solution.
    :return: a ProblemInstance object, along with a constraint-based oracle
    """
    # Create a dictionary with the parameters
    parameters = {"grid_size": 5, "num_categories": 5}

    # Variables
    # Flattened array with 25 elements, representing 5 elements for each of the 5 categories
    grid = cp.intvar(1, 5, shape=(5, 5), name="grid")

    C_T = list()

    # Extract variables for readability
    ukr, norge, eng, spain, jap = grid[0, :]  # Nationalities
    red, blue, yellow, green, ivory = grid[1,:]  # Colors
    oldGold, parly, kools, lucky, chest = grid[2,:]  # Cigarettes
    zebra, dog, horse, fox, snails = grid[3,:]  # Pets
    coffee, tea, h2o, milk, oj = grid[4,:]  # Drinks

    # Add all constraints
    C_T += [(eng == red)]  # Englishman lives in the red house
    C_T += [(spain == dog)]  # Spaniard owns the dog
    C_T += [(coffee == green)]  # Coffee is drunk in the green house
    C_T += [(ukr == tea)]  # Ukrainian drinks tea
    C_T += [(green == ivory + 1)]  # Green house is immediately right of the ivory house
    C_T += [(oldGold == snails)]  # OldGold smoker owns snails
    C_T += [(kools == yellow)]  # Kools are smoked in the yellow house
    C_T += [(milk == 3)]  # Milk is drunk in the middle house
    C_T += [(norge == 1)]  # Norwegian lives in the first house
    C_T += [(abs(chest - fox) == 1)]  # Chesterfield smoker lives next to the man with the fox
    C_T += [(abs(kools - horse) == 1)]  # Kools are smoked in the house next to the house with the horse
    C_T += [(lucky == oj)]  # Lucky smoker drinks orange juice
    C_T += [(jap == parly)]  # Japanese smokes Parliaments
    C_T += [(abs(norge - blue) == 1)]  # Norwegian lives next to the blue house

    # Each row must have different values
    for row in grid:
        C_T += list(cp.AllDifferent(row).decompose())

    # Create the language:
    AV = absvar(2)  # create abstract vars - as many as maximum arity

    # create abstract relations using the abstract vars
    lang = [AV[0] == AV[1], AV[0] != AV[1], AV[0] < AV[1], AV[0] > AV[1], AV[0] >= AV[1], AV[0] <= AV[1],
            abs(AV[0] - AV[1]) == 1, abs(AV[0] - AV[1]) != 1, AV[0] - AV[1] == 1, AV[1] - AV[0] == 1] + [AV[0] == constant for constant in range(1, 6)] + [AV[0] != constant for constant in range(1, 6)]

    instance = ProblemInstance(variables=grid, params=parameters, language=lang, name="zebra")

    oracle = ConstraintOracle(list(set(toplevel_list(C_T))))



    return instance, oracle
