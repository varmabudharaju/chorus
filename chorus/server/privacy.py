"""Deprecated: import from chorus.privacy instead.

This module is preserved as a re-export of chorus.privacy.mechanism for
backward compatibility. It will be removed in v0.3.0.
"""

import warnings

warnings.warn(
    "chorus.server.privacy is deprecated; import from chorus.privacy instead. "
    "This shim will be removed in v0.3.0.",
    DeprecationWarning,
    stacklevel=2,
)

from chorus.privacy.mechanism import GaussianMechanism, apply_dp, clip_delta  # noqa: E402,F401

__all__ = ["GaussianMechanism", "apply_dp", "clip_delta"]
