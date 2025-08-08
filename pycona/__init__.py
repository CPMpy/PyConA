"""
This module initializes the package and imports the necessary components.

Modules:
    ca_environment: Contains the CA environment classes. The core of CA systems
    active_algorithms: Contains the various top-level ICA active_algorithms implemented in the package.
    query_generation: Contains query generation methods.
    find_scope: Contains functions and classes for finding the scope of constraints.
    find_constraint: Contains functions and classes for finding the constraints in the given scopes.
    predictor: PredictorTemplate for predictors to be used during CA. Default is a scikit-learn predictor.
               The file also contains the implementation of a custom Counting based predictor and custom
               feature representations of constraints
    problem_instance: Contains the ProblemInstance class. Containing all information for the instance to be given
                      to the CA system
    answering_queries: Contains the oracles that can be used for query answering in interactive CA. Default is the user.
    metrics: Contains the Metrics class, that is used for logging information.
Helper:
    utils: Contains utility functions used across submodules.
"""

__version__ = "0.3"

from .metrics import Metrics
from .answering_queries import ConstraintOracle, UserOracle
from .ca_environment import ActiveCAEnv, ProbaActiveCAEnv
from .find_constraint import FindC, FindC2
from .query_generation import QGen, TQGen, PQGen, PQGenSolve
from .find_scope import FindScope, FindScope2
from .active_algorithms import QuAcq, PQuAcq, MineAcq, GrowAcq, MQuAcq, MQuAcq2, GenAcq, QuAcqSolve
from .problem_instance import ProblemInstance, absvar, langBasic, langDist, langEqNeq
from .predictor import CountsPredictor, FeaturesRelDim, FeaturesSimpleRel

from .utils import Objectives
