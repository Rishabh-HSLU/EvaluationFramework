"""Shared pytest configuration.

Caps torch's CPU intra-op thread pool. This sandbox exposes 128 logical
CPUs; torch defaults to using all of them as intra-op threads, and for the
small tensors in SBBTS's tests (and generally any CPU-only unit test) that
is a severe anti-pattern -- thread-spawn/synchronization overhead per op
dominates actual compute, turning a millisecond-scale forward pass into a
~12-second one. GPU runs (the actual target hardware for training) are
unaffected by this setting.
"""

from __future__ import annotations

import torch

torch.set_num_threads(4)
