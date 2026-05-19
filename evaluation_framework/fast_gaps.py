"""
FFT-batched B2/B3 gap estimators — same estimand as the bucket classes,
faster at N=200. Verified against bucket compute_gap at startup in bench.
"""

from __future__ import annotations

import numpy as np

from .buckets.b2_nonlinear_temporal import BucketNonlinearTemporal
from .buckets.b3_leverage_effect import BucketLeverageEffect

B2_K_MIN = 60
B2_K_MAX = 390
B3_K_MIN = 1
B3_K_MAX = 390


def _acf_abs_fft(paths: np.ndarray, k_max: int) -> np.ndarray:
    a = np.abs(paths)
    a = a - a.mean(axis=1, keepdims=True)
    T = a.shape[1]
    n_fft = 2 * T
    F = np.fft.rfft(a, n=n_fft, axis=1)
    power = (F * F.conj()).real
    acf = np.fft.irfft(power, n=n_fft, axis=1)
    denom = np.clip(acf[:, 0:1], 1e-12, None)
    return acf[:, 1 : k_max + 1] / denom


def b2_gap(
    real: np.ndarray,
    syn: np.ndarray,
    k_min: int = B2_K_MIN,
    k_max: int = B2_K_MAX,
) -> float:
    acf_r = _acf_abs_fft(real, k_max).mean(axis=0)
    acf_s = _acf_abs_fft(syn, k_max).mean(axis=0)
    diff = np.abs(acf_r[k_min - 1 : k_max] - acf_s[k_min - 1 : k_max])
    return float(diff.mean())


def _leverage_fft(paths: np.ndarray, k_max: int) -> np.ndarray:
    r = paths - paths.mean(axis=1, keepdims=True)
    a = np.abs(paths)
    a = a - a.mean(axis=1, keepdims=True)
    T = paths.shape[1]
    n_fft = 2 * T
    F_r = np.fft.rfft(r, n=n_fft, axis=1)
    F_a = np.fft.rfft(a, n=n_fft, axis=1)
    cross = F_r.conj() * F_a
    xcorr = np.fft.irfft(cross, n=n_fft, axis=1).real
    norm_r = np.sqrt((r * r).sum(axis=1, keepdims=True))
    norm_a = np.sqrt((a * a).sum(axis=1, keepdims=True))
    denom = np.clip(norm_r * norm_a, 1e-12, None)
    return xcorr[:, 1 : k_max + 1] / denom


def b3_gap(
    real: np.ndarray,
    syn: np.ndarray,
    k_min: int = B3_K_MIN,
    k_max: int = B3_K_MAX,
) -> float:
    lev_r = _leverage_fft(real, k_max).mean(axis=0)
    lev_s = _leverage_fft(syn, k_max).mean(axis=0)
    diff = np.abs(lev_r[k_min - 1 : k_max] - lev_s[k_min - 1 : k_max])
    return float(diff.mean())


def verify_fft_vs_bucket(
    real: np.ndarray,
    syn: np.ndarray,
    tol: float = 1e-10,
) -> None:
    r, s = real[:20], syn[:20]
    b2_ref = BucketNonlinearTemporal(k_min=B2_K_MIN, k_max=B2_K_MAX).compute_gap(r, s)
    b2_fft = b2_gap(r, s)
    assert abs(b2_fft - b2_ref) < tol, f"B2 FFT mismatch: {abs(b2_fft - b2_ref)}"
    b3_ref = BucketLeverageEffect(k_min=B3_K_MIN, k_max=B3_K_MAX).compute_gap(r, s)
    b3_fft = b3_gap(r, s)
    assert abs(b3_fft - b3_ref) < tol, f"B3 FFT mismatch: {abs(b3_fft - b3_ref)}"
