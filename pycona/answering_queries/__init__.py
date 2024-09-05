"""
Initializes the oracle module, including different types of oracles:
 - Oracle: Abstract class for oracle implementation
 - ConstraintOracle: Oracle based on the target set of constraints, which is used to answer the given queries
 - UserOracle: The Oracle is a human user, who directly answers the given queries
"""

from .oracle import Oracle
from .constraint_oracle import ConstraintOracle
from .user_oracle import UserOracle
