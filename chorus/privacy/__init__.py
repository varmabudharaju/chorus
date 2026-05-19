"""Privacy primitives and accounting for federated LoRA.

Houses the Gaussian DP mechanism (in mechanism.py) and the stateful
privacy accountant (in accountant.py).
"""

from chorus.privacy.mechanism import GaussianMechanism, apply_dp, clip_delta

try:
    from chorus.privacy.accountant import PrivacyAccountant
    _ACCOUNTANT_AVAILABLE = True
except ImportError:
    _ACCOUNTANT_AVAILABLE = False

__all__ = [
    "PrivacyAccountant",
    "GaussianMechanism",
    "apply_dp",
    "clip_delta",
]
