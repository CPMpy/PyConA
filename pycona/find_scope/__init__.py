"""
Initializes this module, with implementations of different methods from the literature for the FindScope subcomponent:
    - FindScopeBase: Abstract findscope class.
    - FindScope: FindScope function from:
                 Bessiere, Christian, et al., "Constraint acquisition via Partial Queries", IJCAI 2013.
    - FindScope2: FindScope function from:
                  Bessiere, Christian, et al., "Learning constraints through partial queries", AIJ 2023.
"""

from .findscope_core import FindScopeBase
from .findscope import FindScope
from .findscope2 import FindScope2
