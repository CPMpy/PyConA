import cpmpy as cp
from cpmpy.transformations.normalize import toplevel_list
from ..answering_queries.constraint_oracle import ConstraintOracle
from ..problem_instance import ProblemInstance, absvar



def construct_nqueens_problem(n):

    parameters = {"n": n}

    queens = cp.intvar(1, n, shape=n, name="queens")

    # Model
    model = cp.Model()

    # Constraints list
    CT = []

    CT += list(cp.AllDifferent(queens).decompose())

    for i in range(n):
        for j in range(i + 1, n): # Compare each queen with every other queen once
            CT += [(queens[i] - i != queens[j] - j)]  # Different major diagonals
            CT += [(queens[i] + i != queens[j] + j)]  # Different minor diagonals


    # Add all collected constraints to the model
    model += CT

    C_T = toplevel_list(CT) 

    AV = absvar(2)
    #lang = [AV[0] == AV[1], AV[0] != AV[1], AV[0] < AV[1], AV[0] > AV[1], AV[0] >= AV[1], AV[0] <= AV[1]] + 
    lang = [AV[0] - AV[1] == constant for constant in range(-n, 2*n)] + [AV[0] - AV[1] != constant for constant in range(-n, 2*n)]

    instance = ProblemInstance(variables=queens, params=parameters, language=lang, name="nqueens")

    oracle = ConstraintOracle(list(set(toplevel_list(C_T))))

    print("oracle constraints: ", len(oracle.constraints))
    for c in oracle.constraints:
        print(c)

    input("Press Enter to continue...")

    return instance, oracle
