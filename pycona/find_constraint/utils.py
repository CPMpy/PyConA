from itertools import chain


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
    C3 = [[c1 & c2 if c1 is not c2 else c1 for c2 in C2] for c1 in C1]
    C3 = list(chain.from_iterable(C3))
    C3 = remove_redundant_conj(C3)
    return C3


def remove_redundant_conj(C1):
    """
    Remove redundant conjunctions from the given list of constraints.

    :param C1: A list of constraints.
    :return: A list of constraints with redundant conjunctions removed.
    """
    C2 = list()

    for c in C1:
        C = [c]
        conj_args = []

        while len(C) > 0:
            c1 = C.pop()

            if c1.name == 'and':
                [C.append(c2) for c2 in c1.args]
            else:
                conj_args.append(c1)

        flag_eq = False
        flag_neq = False
        flag_geq = False
        flag_leq = False
        flag_ge = False
        flag_le = False

        for c1 in conj_args:
            print(c1.name)
            # Tias is on 3.9, no 'match' please!
            if c1.name == "==":
                flag_eq = True
            elif c1.name == "!=":
                flag_neq = True
            elif c1.name == "<=":
                flag_leq = True
            elif c1.name == ">=":
                flag_geq = True
            elif c1.name == "<":
                flag_le = True
            elif c1.name == ">":
                flag_ge = True
            else:
                raise Exception("constraint name is not recognised")

            if not ((flag_eq and (flag_neq or flag_le or flag_ge)) or (
                    (flag_leq or flag_le) and (flag_geq or flag_ge))):
                C2.append(c)
    return C2
