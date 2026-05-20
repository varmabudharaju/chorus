"""Privacy primitives and accounting for federated LoRA.

Houses the Gaussian DP mechanism (in mechanism.py) and the stateful
privacy accountant (in accountant.py).
"""

from chorus.privacy.mechanism import GaussianMechanism, apply_dp, clip_delta

try:
    from chorus.privacy.accountant import PrivacyAccountant
    __all__ = ["PrivacyAccountant", "GaussianMechanism", "apply_dp", "clip_delta"]
except ImportError:
    # PrivacyAccountant requires the [privacy] extra (dp-accounting or opacus).
    # The mechanism remains available without it.
    __all__ = ["GaussianMechanism", "apply_dp", "clip_delta"]
