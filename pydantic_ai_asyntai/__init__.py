"""Pydantic AI capability for Asyntai, the AI chat agent for websites."""

from .capability import INSTRUCTIONS, AsyntaiCapability
from .client import AsyntaiClient, AsyntaiError, AsyntaiRequestError

__all__ = [
    'INSTRUCTIONS',
    'AsyntaiCapability',
    'AsyntaiClient',
    'AsyntaiError',
    'AsyntaiRequestError',
]

__version__ = '0.1.0'
