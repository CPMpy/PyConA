from itertools import chain
import cpmpy as cp


def get_max_conjunction_size(C1):
    """
    Calculate the maximum size of conjunctions in the given list of constraints.

    :param C1: A list of constraints.
    :return: The maximum size of conjunctions.
    """
    max_conj_size = 0

    for c in C1:
        C = [c]
        conj_args = []

        while len(C) > 0:
            c1 = C.pop()

            if c1.name == 'and':
                [C.append(c2) for c2 in c1.args]
            else:
                conj_args.append(c1)

        max_conj_size = max(len(conj_args), max_conj_size)

    return max_conj_size


def get_delta_p(C1):
    """
    Generate a list of lists of constraints grouped by the size of their conjunctions.

    :param C1: A list of constraints.
    :return: A list of lists of constraints grouped by conjunction size.
    """
    max_conj_size = get_max_conjunction_size(C1)

    Delta_p = [[] for _ in range(max_conj_size)]

    for c in C1:

        C = [c]
        conj_args = []

        while len(C) > 0:
            c1 = C.pop()

            if c1.name == 'and':
                [C.append(c2) for c2 in c1.args]
            else:
                conj_args.append(c1)

        Delta_p[len(conj_args) - 1].append(c)

    return Delta_p


def join_con_net(C1, C2):
    """
    Join two lists of constraints into a single list by performing conjunctions.

    :param C1: The first list of constraints.
    :param C2: The second list of constraints.
    :return: A list of constraints resulting from the conjunction of C1 and C2.
    """
    C3 = [[set(c1 + c2) for c2 in unravel_conjunctions(C2)] for c1 in unravel_conjunctions(C1)]
    C3 = list(chain.from_iterable(C3))
    C3 = [cp.all(c) for c in C3]
    C3 = remove_redundant_conj(C3)
    return C3


def get_conjunction_args(constraint):
    """
    Break down a constraint into its constituent conjunctive arguments.
    
    Args:
        constraint: A CPMpy constraint that may contain conjunctions
        
    Returns:
        list: A list of atomic constraints that make up the conjunction
    """
    stack = [constraint]
    conj_args = []

    while stack:
        current = stack.pop()
        if current.name == 'and':
            stack.extend(current.args)
        else:
            conj_args.append(current)
            
    return conj_args


def remove_redundant_conj(constraints: list) -> list:
    """
    Remove redundant conjunctions from the given list of constraints.
    A conjunction is considered redundant if:
    1. It contains the same set of atomic constraints as another conjunction, or
    2. It is unsatisfiable
    
    Args:
        constraints: A list of CPMpy constraints, potentially containing conjunctions
        
    Returns:
        list: A filtered list of constraints with redundant conjunctions removed
        
    Example:
        >>> x = cp.intvar(0, 10, "x")
        >>> constraints = [x >= 0, x >= 0 & x <= 5, x >= 2 & x <= 5]
        >>> result = remove_redundant_conj(constraints)
        >>> len(result) < len(constraints)  # Some redundant constraints removed
        True
    """
    unique_constraints = []
    unique_atomic_sets = []
    
    for constraint in constraints:
        # Break down the constraint into atomic parts
        atomic_constraints = get_conjunction_args(constraint)
        
        # Check if this set of atomic constraints is unique
        is_redundant = any(
            len(atomic_constraints) == len(existing_set) and 
            set(atomic_constraints) == set(existing_set)
            for existing_set in unique_atomic_sets
        )
        
        if not is_redundant:
            # Verify the constraint is satisfiable
            try:
                if cp.Model(constraint).solve():
                    unique_constraints.append(constraint)
                    unique_atomic_sets.append(atomic_constraints)
            except cp.exceptions.UnsatisfiableError:
                # Skip unsatisfiable constraints
                continue
                
    return unique_constraints

def unravel_conjunctions(constraints: list) -> list:
    """
    Unravel conjunctions in the given list of constraints.
    """
    if not isinstance(constraints, list):
        constraints = [constraints]

    unraveled = []
    for c in constraints:
        if c.name == 'and':
            sub_list = []
            for sub_c in c.args:
                sub_list.append(sub_c)
            unraveled.append(sub_list)
        else:
            unraveled.append([c])

    return unraveled