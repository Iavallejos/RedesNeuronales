import unittest
from perceptron import Perceptron

case1 = [0, 0]
case2 = [1, 0]
case3 = [0, 1]
case4 = [1, 1]


class TestAND(unittest.TestCase):
    andPerceptron = Perceptron([-3, 2, 2])

    def test_1(self):
        self.assertEqual(self.andPerceptron.calculate(case1), 0)

    def test_2(self):
        self.assertEqual(self.andPerceptron.calculate(case2), 0)

    def test_3(self):
        self.assertEqual(self.andPerceptron.calculate(case3), 0)

    def test_4(self):
        self.assertEqual(self.andPerceptron.calculate(case4), 1)


class TestNAND(unittest.TestCase):
    nandPerceptron = Perceptron([3, -2, -2])

    def test_1(self):
        self.assertEqual(self.nandPerceptron.calculate(case1), 1)

    def test_2(self):
        self.assertEqual(self.nandPerceptron.calculate(case2), 1)

    def test_3(self):
        self.assertEqual(self.nandPerceptron.calculate(case3), 1)

    def test_4(self):
        self.assertEqual(self.nandPerceptron.calculate(case4), 0)


class TestOR(unittest.TestCase):
    orPerceptron = Perceptron([0, 1, 1])

    def test_1(self):
        self.assertEqual(self.orPerceptron.calculate(case1), 0)

    def test_2(self):
        self.assertEqual(self.orPerceptron.calculate(case2), 1)

    def test_3(self):
        self.assertEqual(self.orPerceptron.calculate(case3), 1)

    def test_4(self):
        self.assertEqual(self.orPerceptron.calculate(case4), 1)


if __name__ == '__main__':
    unittest.main()
