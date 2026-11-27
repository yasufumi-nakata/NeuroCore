"""Xavier inference services with lazy hardware backends.

Forecasts and explicit-goal action proposals are engineering capabilities.
They do not establish independently evaluated EEG intent inference.
"""

from .registry import ModelRegistry
from .service import XavierService, XavierServiceError

__all__ = ["ModelRegistry", "XavierService", "XavierServiceError"]
