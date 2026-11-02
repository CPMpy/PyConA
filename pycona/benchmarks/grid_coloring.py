import cpmpy as cp
from cpmpy.transformations.normalize import toplevel_list
from ..answering_queries.constraint_oracle import ConstraintOracle
from ..problem_instance import ProblemInstance, absvar


def _chebyshev_dist(i1, j1, i2, j2):
    """Chebyshev (L∞) distance: max(|Δi|, |Δj|)."""
    return max(abs(i1 - i2), abs(j1 - j2))


def _manhattan_dist(i1, j1, i2, j2):
    """Manhattan (L1) distance: |Δi| + |Δj|."""
    return abs(i1 - i2) + abs(j1 - j2)


def construct_grid_coloring(rows=10, cols=10, num_colors=18, max_distance=2, distance_metric="chebyshev"):
    """
    :Description: A distance-based grid coloring problem. Each cell of the grid gets a color from 1..num_colors,
    so that any two cells within max_distance of each other have different colors.

    Distance metrics:
    - "chebyshev" (L∞): dist = max(|Δi|, |Δj|). max_distance=1 gives 8-neighbors.
    - "manhattan" (L1): dist = |Δi| + |Δj|. max_distance=1 gives 4-neighbors.

    :param rows: number of rows in the grid
    :param cols: number of columns in the grid
    :param num_colors: number of colors (domain size), at least 2
    :param max_distance: maximum distance for the "different color" constraint
    :param distance_metric: "chebyshev" or "manhattan"
    :return: a ProblemInstance object, along with a constraint-based oracle
    """
    if num_colors < 2:
        raise ValueError("num_colors must be at least 2")
    if max_distance < 1:
        raise ValueError("max_distance must be at least 1")
    if distance_metric not in ("chebyshev", "manhattan"):
        raise ValueError("distance_metric must be 'chebyshev' or 'manhattan'")

    dist_fn = _chebyshev_dist if distance_metric == "chebyshev" else _manhattan_dist

    # Create a dictionary with the parameters
    parameters = {
        "rows": rows,
        "cols": cols,
        "num_colors": num_colors,
        "max_distance": max_distance,
        "distance_metric": distance_metric,
    }

    # grid[i, j] is the color assigned to cell (i, j)
    grid = cp.intvar(1, num_colors, shape=(rows, cols), name="grid")

    model = cp.Model()

    # Cells within max_distance must have different colors
    for i1 in range(rows):
        for j1 in range(cols):
            for i2 in range(rows):
                for j2 in range(cols):
                    if (i1, j1) >= (i2, j2):
                        continue
                    if dist_fn(i1, j1, i2, j2) <= max_distance:
                        model += grid[i1, j1] != grid[i2, j2]

    if not model.solve():
        raise Exception("The problem has no solution")

    C_T = list(set(toplevel_list(model.constraints)))

    # Create the language:
    AV = absvar(2)  # create abstract vars - as many as maximum arity

    # create abstract relations using the abstract vars
    lang = [AV[0] == AV[1], AV[0] != AV[1], AV[0] < AV[1], AV[0] > AV[1], AV[0] >= AV[1], AV[0] <= AV[1]]

    instance = ProblemInstance(
        variables=grid,
        params=parameters,
        language=lang,
        name=f"grid_coloring_{rows}x{cols}_colors{num_colors}_d{max_distance}_{distance_metric}",
    )

    oracle = ConstraintOracle(C_T)

    return instance, oracle
