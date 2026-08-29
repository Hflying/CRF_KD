"""Minimal koi.decode stub.

Used by bonito.io (sometimes). We don't call beam_search in the CRF-KD
training pipeline (Viterbi only), so a stub that errors on call is fine.
"""


def beam_search(*args, **kwargs):
    raise NotImplementedError("koi.decode.beam_search is not available (koi stub)")


def to_str(*args, **kwargs):
    raise NotImplementedError("koi.decode.to_str is not available (koi stub)")
