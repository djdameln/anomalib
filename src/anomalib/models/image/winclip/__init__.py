"""WinCLIP Model for anomaly detection.

This module implements anomaly detection using the WinCLIP model, which leverages
CLIP embeddings and a sliding window approach to detect anomalies in images.

Example:
    >>> from anomalib.models.image import WinClip
    >>> model = WinClip()  # doctest: +SKIP
    >>> model.fit(["normal1.jpg", "normal2.jpg"])  # doctest: +SKIP
    >>> prediction = model.predict("test.jpg")  # doctest: +SKIP

See Also:
    - :class:`WinClip`: Main model class for WinCLIP-based anomaly detection
    - :class:`WinClipModel`: PyTorch implementation of the WinCLIP model
"""

# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .lightning_model import WinClip
from .torch_model import WinClipModel

__all__ = ["WinClip", "WinClipModel"]
