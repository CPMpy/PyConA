"""
Initializes this module, with implementations of different methods from the literature for the FindC subcomponent:
 - FindCBase: Abstract FindC class
 - FindC: FindC function from  Bessiere, Christian, et al., "Constraint acquisition via Partial Queries", IJCAI 2013.
 - FindC2: FindC function from Bessiere, Christian, et al., "Learning constraints through partial queries", AIJ 2023
"""

from .findc import FindC
from .findc_core import FindCBase
from .findc2 import FindC2
