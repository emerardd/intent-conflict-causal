import numpy as np

from intent_conflict.stage2_synthesis import _cosine, _pearson


def test_cosine_handles_identical_and_orthogonal_vectors() -> None:
    assert np.isclose(_cosine(np.asarray([1.0, 2.0]), np.asarray([1.0, 2.0])), 1.0)
    assert np.isclose(_cosine(np.asarray([1.0, 0.0]), np.asarray([0.0, 2.0])), 0.0)


def test_pearson_preserves_linear_precision_agreement() -> None:
    low_precision = np.asarray([-2.0, -1.0, 1.0, 3.0])
    high_precision = 2.5 * low_precision + 0.75
    assert np.isclose(_pearson(low_precision, high_precision), 1.0)
