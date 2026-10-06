"""The spring (coil) element.

Renamed from "helix" for clarity (helix and thread are separate elements in
this generator). This module re-exports the implementation, which currently
still lives in ``sweep_init`` for back-compat; new code should import from here.
"""
from .sweep_init import (  # noqa: F401
    spring,
    Spring,
    SpringFactory,
    SweepInit,
    SweepInitFactory,
)
