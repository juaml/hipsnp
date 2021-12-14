import numpy as np
from pathlib import Path
from pandas._testing import assert_frame_equal


def assert_genotype_equal(expected, actual):
    """
    Asserts that two genotypes are equal.
    """
    assert_frame_equal(expected.metadata, actual.metadata)
    assert expected.probabilities.keys() == actual.probabilities.keys()

    assert all(
        [np.array_equal(a[1], b[1], equal_nan=True)
         for a, b in zip(expected.probabilities.values(),
                         actual.probabilities.values())])


def get_testing_data_dir():
    t_path = Path(__file__).parent.parent / 'tests' / 'data'

    return t_path
