import cpmpy as cp

from ..answering_queries.constraint_oracle import ConstraintOracle
from ..problem_instance import ProblemInstance, absvar
from cpmpy.transformations.normalize import toplevel_list

def day_of_exam(course, slots_per_day):
    return course // slots_per_day


def construct_examtt_simple(nsemesters=6, courses_per_semester=3, slots_per_day=3, days_for_exams=10):
    """
    :return: a ProblemInstance object, along with a constraint-based oracle
    """
    total_slots = slots_per_day * days_for_exams

    parameters = {'nsemesters': nsemesters, 'courses_per_semester': courses_per_semester,
                      'slots_per_day': slots_per_day, 'days_for_exams': days_for_exams}

    # Variables
    courses = cp.intvar(1, total_slots, shape=(nsemesters, courses_per_semester), name="courses")

    model = cp.Model()
    model += cp.AllDifferent(courses).decompose()

    # Constraints on courses of same semester
    for row in courses:
        model += cp.AllDifferent(day_of_exam(row, slots_per_day)).decompose()

    C_T = list(set(toplevel_list(model.constraints)))

    if model.solve():
        courses.clear()
    else:
        print("no solution")

    # Create the language:
    AV = absvar(2)  # create abstract vars - as many as maximum arity

    # create abstract relations using the abstract vars
    lang = [AV[0] == AV[1], AV[0] != AV[1], AV[0] < AV[1], AV[0] > AV[1], AV[0] >= AV[1], AV[0] <= AV[1],
            day_of_exam(AV[0], slots_per_day) != day_of_exam(AV[1], slots_per_day),
            day_of_exam(AV[0], slots_per_day) == day_of_exam(AV[1], slots_per_day)]

    instance = ProblemInstance(variables=courses, params=parameters, language=lang,
                               name=f"exam_timetabling_semesters{nsemesters}_courses{courses_per_semester}_"
                                    f"timeslots{slots_per_day}_days{days_for_exams}")

    oracle = ConstraintOracle(C_T)

    return instance, oracle
