import cpmpy as cp
from cpmpy.transformations.normalize import toplevel_list
from ..answering_queries.constraint_oracle import ConstraintOracle
from ..problem_instance import ProblemInstance, absvar


def construct_gtsudoku(block_size_row=2, block_size_col=2, grid_size=4):
    """
    :return: a ProblemInstance object, along with a constraint-based oracle
    """

    # Create a dictionary with the parameters
    parameters = {"block_size_row": block_size_row, "block_size_col": block_size_col, "grid_size": grid_size}

    # Variables
    grid = cp.intvar(1, grid_size, shape=(grid_size, grid_size), name="grid")

    model = cp.Model()

    # Constraints on rows and columns
    for row in grid:
        model += cp.AllDifferent(row).decompose()

    for col in grid.T:  # numpy's Transpose
        model += cp.AllDifferent(col).decompose()

    # Constraints on blocks
    for i in range(0, grid_size, block_size_row):
        for j in range(0, grid_size, block_size_col):
            model += cp.AllDifferent(grid[i:i + block_size_row, j:j + block_size_col]).decompose()  # python's indexing

    true_horizontal_gt = [
        (0, 0, 0, 1),  
        (1, 1, 1, 2),  
        (2, 2, 2, 3),  
        (3, 3, 3, 4),  
        (4, 4, 4, 5),  
    ]
    
    for r1, c1, r2, c2 in true_horizontal_gt:
        if r2 < grid_size and c2 < grid_size:
            model += (grid[r1, c1] > grid[r2, c2])
            
    true_vertical_gt = [
        (0, 2, 1, 2),  
        (1, 3, 2, 3),  
        (2, 4, 3, 4),  
        (3, 5, 4, 5),  
        (4, 6, 5, 6),  
    ]
    
    for r1, c1, r2, c2 in true_vertical_gt:
        if r1 < grid_size and r2 < grid_size and c1 < grid_size and c2 < grid_size:
            model += (grid[r1, c1] > grid[r2, c2])


    C_T = list(set(toplevel_list(model.constraints)))

    # Create the language:
    AV = absvar(2)  # create abstract vars - as many as maximum arity

    # create abstract relations using the abstract vars
    lang = [AV[0] == AV[1], AV[0] != AV[1], AV[0] < AV[1], AV[0] > AV[1], AV[0] >= AV[1], AV[0] <= AV[1]]

    instance = ProblemInstance(variables=grid, params=parameters, language=lang, name=f"sudoku_{block_size_row}_{block_size_col}_{grid_size}")

    oracle = ConstraintOracle(C_T)

    return instance, oracle
