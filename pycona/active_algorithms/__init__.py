"""
This module imports various active_algorithms for ICA implementations:
 - QuAcq:
 - MQuAcq:
 - MQuAcq2:
 - GrowAcq:
"""

from .algorithm_core import AlgorithmCAInteractive
from .quacq import QuAcq
from .mquacq2 import MQuAcq2
from .mquacq import MQuAcq
from .growacq import GrowAcq
from .pquacq import PQuAcq
from .mineacq import MineAcq
from .genacq import GenAcq
