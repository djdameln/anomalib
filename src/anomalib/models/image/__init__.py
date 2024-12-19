"""Anomalib Image Models.

This module contains implementations of various deep learning models for image-based
anomaly detection.

Example:
    >>> from anomalib.models.image import Padim, Patchcore
    >>> # Initialize a model
    >>> model = Padim()  # doctest: +SKIP
    >>> # Train on normal images
    >>> model.fit(["normal1.jpg", "normal2.jpg"])  # doctest: +SKIP
    >>> # Get predictions
    >>> predictions = model.predict("test.jpg")  # doctest: +SKIP

Available Models:
    - :class:`Cfa`: Contrastive Feature Aggregation
    - :class:`Cflow`: Conditional Normalizing Flow
    - :class:`Csflow`: Conditional Split Flow
    - :class:`Dfkde`: Deep Feature Kernel Density Estimation
    - :class:`Dfm`: Deep Feature Modeling
    - :class:`Draem`: Dual Reconstruction by Adversarial Masking
    - :class:`Dsr`: Deep Spatial Reconstruction
    - :class:`EfficientAd`: Efficient Anomaly Detection
    - :class:`Fastflow`: Fast Flow
    - :class:`Fre`: Feature Reconstruction Error
    - :class:`Ganomaly`: Generative Adversarial Networks
    - :class:`Padim`: Patch Distribution Modeling
    - :class:`Patchcore`: Patch Core
    - :class:`ReverseDistillation`: Reverse Knowledge Distillation
    - :class:`Stfpm`: Student-Teacher Feature Pyramid Matching
    - :class:`Uflow`: Unsupervised Flow
    - :class:`VlmAd`: Vision Language Model Anomaly Detection
    - :class:`WinClip`: Zero-/Few-Shot CLIP-based Detection
"""

# Copyright (C) 2023-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .cfa import Cfa
from .cflow import Cflow
from .csflow import Csflow
from .dfkde import Dfkde
from .dfm import Dfm
from .draem import Draem
from .dsr import Dsr
from .efficient_ad import EfficientAd
from .fastflow import Fastflow
from .fre import Fre
from .ganomaly import Ganomaly
from .padim import Padim
from .patchcore import Patchcore
from .reverse_distillation import ReverseDistillation
from .stfpm import Stfpm
from .uflow import Uflow
from .vlm_ad import VlmAd
from .winclip import WinClip

__all__ = [
    "Cfa",
    "Cflow",
    "Csflow",
    "Dfkde",
    "Dfm",
    "Draem",
    "Dsr",
    "EfficientAd",
    "Fastflow",
    "Fre",
    "Ganomaly",
    "Padim",
    "Patchcore",
    "ReverseDistillation",
    "Stfpm",
    "Uflow",
    "VlmAd",
    "WinClip",
]
