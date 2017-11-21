import unittest
import numpy as np
from claude.measures import information_entropy


class TestEntropy(unittest.TestCase):
    SAMPLE_SIZE = 10000
    SAMPLE_SIZE_SMALL = 100
    TOLERANCE = 1e-3

    def within_tolerance(self, value1, value2):
        return np.abs(value1 - value2) < self.TOLERANCE

    def test_zero_entropy(self):
        x = np.ones(self.SAMPLE_SIZE_SMALL)
        self.assertEqual(information_entropy(x), 0)

    def test_max_entropy_1(self):
        x0 = np.zeros(self.SAMPLE_SIZE_SMALL)
        x1 = np.ones(self.SAMPLE_SIZE_SMALL)
        x = np.concatenate((x0, x1))
        self.assertEqual(information_entropy(x), 1)

    def test_max_entropy_2(self):
        x = np.concatenate([i * np.ones(self.SAMPLE_SIZE_SMALL)
                            for i in range(4)])
        self.assertEqual(information_entropy(x), 2)

    def test_entropy_1(self):
        x = np.random.random(self.SAMPLE_SIZE) > 0.5
        self.assertEqual(self.within_tolerance(information_entropy(x), 1),
                         True)

    def test_entropy_2(self):
        sample_size = 10
        x = np.arange(sample_size)
        self.assertEqual(self.within_tolerance(information_entropy(x),
                                               np.log2(sample_size)), True)


if __name__ == '__main__':
    unittest.main()
