import cpmpy as cp

from pycona.answering_queries.constraint_oracle import ConstraintOracle
from pycona.problem_instance import ProblemInstance, absvar

from pycona.utils import combine_sets_distinct, get_combinations, combine_multiple_sets


def construct_golomb(marks=4):
    """
    :return: a ProblemInstance object, along with a constraint-based oracle
    """

    # Create a dictionary with the parameters
    parameters = {"marks": marks}

    # Variables
    marks = cp.intvar(1, 10, shape=(1, marks), name="grid")

    model = cp.Model()

    combinations = get_combinations(marks, 2)
    sets = [combinations, combinations]
    result_combinations = combine_multiple_sets([combinations, combinations])
    for ((v1, v2), (v3, v4)) in result_combinations:
        model += abs(v1 - v2) != abs(v3 - v4)

    C_T = list(model.constraints)
    print("len cT" , len(C_T))
    # Create the language:
    AV = absvar(4)   # create abstract vars - as many as maximum arity

    # create abstract relations using the abstract vars
    lang = [AV[0] == AV[1], AV[0] != AV[1], AV[0] < AV[1], AV[0] > AV[1], AV[0] >= AV[1], AV[0] <= AV[1],
            abs(AV[0] - AV[1]) != abs(AV[2] - AV[3])
            ] # Include different permutations for the distances to cover all

    instance = ProblemInstance(variables=marks, params=parameters, language=lang, name="golomb")
    oracle = ConstraintOracle(C_T)

    return instance, oracle
