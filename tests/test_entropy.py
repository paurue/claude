import pytest
import numpy as np
from claude.information import information_entropy


SAMPLE_SIZE = 10000
SAMPLE_SIZE_SMALL = 100


def within_tolerance(value1, value2):
    TOLERANCE = 1e-3
    return np.abs(value1 - value2) < TOLERANCE


def test_zero_entropy():
    x = np.ones(SAMPLE_SIZE_SMALL)
    assert information_entropy(x) == 0


def test_max_entropy_1():
    x0 = np.zeros(SAMPLE_SIZE_SMALL)
    x1 = np.ones(SAMPLE_SIZE_SMALL)
    x = np.concatenate((x0, x1))
    assert information_entropy(x) == 1


def test_max_entropy_2():
    x = np.concatenate([i * np.ones(SAMPLE_SIZE_SMALL)
                        for i in range(4)])
    assert information_entropy(x) == 2


def test_entropy_1():
    x = np.random.random(SAMPLE_SIZE) > 0.5
    assert within_tolerance(information_entropy(x), 1)


def test_entropy_2():
    sample_size = 10
    x = np.arange(sample_size)
    assert within_tolerance(information_entropy(x),
                            np.log2(sample_size))
