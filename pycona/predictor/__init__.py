"""
Initializes this module, with implementations of different predictors and feature representations of constraints:
    - CountsPredictor: Counting based predictor from:
    Dimos Tsouros, Senne Berden, and Tias Guns. "Guided Bottom-Up Interactive Constraint Acquisition." CP, 2023
    - FeaturesSimpleRel: Simple feature representation storing only the constraint relation, from:
    Dimos Tsouros, Senne Berden, and Tias Guns. "Guided Bottom-Up Interactive Constraint Acquisition." CP, 2023
    - FeaturesRelDim: Feature representation that can be used with any classifier, from:
    Dimos Tsouros, Senne Berden, and Tias Guns. "Learning to Learn in Interactive Constraint Acquisition." AAAI, 2024
"""

from .predictor import CountsPredictor
from .feature_representation import FeaturesSimpleRel, FeaturesRelDim, FeatureRepresentation
