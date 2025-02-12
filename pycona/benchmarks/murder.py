
import cpmpy as cp
from cpmpy.transformations.normalize import toplevel_list
from ..answering_queries.constraint_oracle import ConstraintOracle
from ..problem_instance import ProblemInstance, absvar

def construct_murder_problem():
    """
    :Desciption: Someone was murdered last night, and you are summoned to investigate the murder. The objects found
    on the spot that do not belong to the victim include: a pistol, an umbrella, a cigarette, a diary,
    and a threatening letter. There are also witnesses who testify that someone had argued with the victim,
    someone left the house, someone rang the victim, and some walked past the house several times about the time the
    murder occurred. The suspects are: Miss Linda Ablaze, Mr. Tom Burner, Ms. Lana Curious, Mrs. Suzie Dulles,
    and Mr. Jack Evilson. Each suspect has a different motive for the murder, including: being harassed, abandoned,
    sacked, promotion and hate. Under a set of additional clues given in the description, the problem is who was the
    Murderer? And what was the motive, the evidence-object, and the activity associated with each suspect.
    :return: a ProblemInstance object, along with a constraint-based oracle
    """

    # Variables
    grid = cp.intvar(1, 5, shape=(4, 5), name="grid")

    C_T = list()

    C_T += [cp.AllDifferent(row).decompose() for row in grid]
    # Additional constraints of the murder problem
    C_T += [grid[0, 1] == grid[1, 2]]
    C_T += [grid[0, 2] != grid[1, 4]]
    C_T += [grid[1, 4] != grid[3, 2]]
    C_T += [grid[0, 2] != grid[1, 0]]
    C_T += [grid[0, 2] != grid[3, 4]]
    C_T += [grid[1, 3] == grid[3, 4]]
    C_T += [grid[1, 1] == grid[2, 1]]
    C_T += [grid[0, 3] == grid[2, 3]]
    C_T += [grid[2, 0] == grid[3, 3]]
    C_T += [grid[0, 0] != grid[2, 4]]
    C_T += [grid[0, 0] != grid[1, 4]]
    C_T += [grid[0, 0] == grid[3, 0]]

    # Create the language:
    AV = absvar(2)   # create abstract vars - as many as maximum arity

    # create abstract relations using the abstract vars
    lang = [AV[0] == AV[1], AV[0] != AV[1], AV[0] < AV[1], AV[0] > AV[1], AV[0] >= AV[1], AV[0] <= AV[1]]

    instance = ProblemInstance(variables=grid, language=lang, name="murder")

    oracle = ConstraintOracle(list(set(toplevel_list(C_T))))

    return instance, oracle
