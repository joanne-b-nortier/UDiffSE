"""
Adapted from original code by Clarity Challenge
https://github.com/claritychallenge/clarity
"""
import os
import numpy as np
from pypesq import pesq
from pystoi import stoi
import pyloudnorm as pyln
import sys

sys.path.append(os.path.join(os.getcwd(), "metrics"))
import src.dnnmos_metric.dnnmos_metric as dnnmos_metric
import warnings


import warnings

warnings.filterwarnings("ignore")


meter = pyln.Meter(16000)


def normalize(x, target_loudness=-30, meter=None, sr=16000):
    """
    LUFS normalization of a signal using pyloudnorm.

    Parameters
    ----------
    x : ndarray
        Input signal.
    target_loudness : float, not compulsory
        Target loudness of the output in dB LUFS. The default is -30.
    meter : Meter, not compulsory
        The pyloudnorm BS.1770 meter. The default is None.
    sr : int, not compulsory
        Sampling rate. The default is 16000.

    Returns
    -------
    x_norm : ndarray
        Normalized output signal.
    """

    if meter is None:
        meter = pyln.Meter(sr)  # create BS.1770 meter

    # peak normalize to 0.7 to ensure that the meter does not return -inf
    x = x - np.mean(x)
    x = x / (np.max(np.abs(x)) + 1e-9) * 0.7

    # measure the loudness first
    loudness = meter.integrated_loudness(x)

    # loudness normalize audio to target_loudness LUFS
    x_norm = pyln.normalize.loudness(x, loudness, target_loudness)

    return x_norm


def compute_dnnmos(x, sr):
    """Compute DNN-MOS metric
    Args:
        x: array of float, shape (n_samples,)
        sr (int): sample rate of files
    Returns:
        DNN-MOS metric (dict): SIG_MOS, BAK_MOS, OVR_MOS (float)
    """
    x = normalize(x, target_loudness=-30, meter=meter, sr=sr)

    dnsmos_res = dnnmos_metric.compute_dnsmos(x, fs=sr)

    return dnsmos_res


def compute_sisdr(reference, estimate):
    """Compute the scale invariant SDR.

    Parameters
    ----------
    estimate : array of float, shape (n_samples,)
        Estimated signal.
    reference : array of float, shape (n_samples,)
        Ground-truth reference signal.

    Returns
    -------
    sisdr : float
        SI-SDR.

    Example
    --------
    >>> import numpy as np
    >>> from sisdr_metric import compute_sisdr
    >>> np.random.seed(0)
    >>> reference = np.random.randn(16000)
    >>> estimate = np.random.randn(16000)
    >>> compute_sisdr(estimate, reference)
    -48.1027283264049
    """
    eps = np.finfo(estimate.dtype).eps
    alpha = (np.sum(estimate * reference) + eps) / (
        np.sum(np.abs(reference) ** 2) + eps
    )
    sisdr = 10 * np.log10(
        (np.sum(np.abs(alpha * reference) ** 2) + eps)
        / (np.sum(np.abs(alpha * reference - estimate) ** 2) + eps)
    )
    return sisdr


def compute_pesq(target, enhanced, sr):
    """Compute PESQ using PyPESQ
    Args:
        target (string): Name of file to read
        enhanced (string): Name of file to read
        sr (int): sample rate of files
    Returns:
        PESQ metric (float)
    """
    len_x = np.min([len(target), len(enhanced)])
    target = target[:len_x]
    enhanced = enhanced[:len_x]

    return pesq(target, enhanced, sr)


def compute_stoi(target, enhanced, sr):
    """Compute STOI from: https://github.com/mpariente/pystoi
    Args:
        target (string): Name of file to read
        enhanced (string): Name of file to read
        sr (int): sample rate of files
    Returns:
        STOI metric (float)
    """
    len_x = np.min([len(target), len(enhanced)])
    target = target[:len_x]
    enhanced = enhanced[:len_x]

    return stoi(target, enhanced, sr, extended=True)
