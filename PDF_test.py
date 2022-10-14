import unittest
import sys

import numpy as np
import matplotlib.pyplot as plt

if sys.flags.interactive:
    plt.ion()

from assignment_3 import ProbabilityDensityFunction


class TestPDF(unittest.TestCase):
    def test_uniform(self):
        """Test a uniform distribution from class ProbabilityDensityFunction"""

        x = np.linspace(0., 1., 100)
        y = np.full(x.shape, 1.)
        pdf = ProbabilityDensityFunction(x, y)
        self.assertAlmostEqual(pdf.integral(0., 1.), 1.)
        plt.figure('Uniform pdf')
        plt.plot(x, pdf(x))
        plt.figure('Uniform cdf')
        plt.plot(x, pdf.cdf(x))
        plt.figure('Uniform ppf')
        plt.plot(x, pdf.ppf(x))
        r = pdf.gen_random(1000000)
        plt.figure('Uniform random variate')
        plt.hist(r, bins=100)

    def test_triangular(self):
        """Test a triangular distribution from class ProbabilityDensityFunction"""

        x = np.linspace(0., 1., 100)
        y = 4. * x
        pdf = ProbabilityDensityFunction(x, y)
        self.assertAlmostEqual(pdf.integral(0., 1.), 1.)
        plt.figure('Triangular pdf')
        plt.plot(x, pdf(x))
        plt.figure('Triangular cdf')
        plt.plot(x, pdf.cdf(x))
        plt.figure('Triangular ppf')
        plt.plot(x, pdf.ppf(x))
        r = pdf.gen_random(1000000)
        plt.figure('Triangular random variate')
        plt.hist(r, bins=100)

    def test_exponential(self):
        """Test an exponential distribution from class ProbabilityDensityFunction"""

        x = np.linspace(0., 1., 100)
        y = np.exp(10*x)
        pdf = ProbabilityDensityFunction(x, y)
        print(pdf.integral(0., 1.))
        self.assertAlmostEqual(pdf.integral(0., 1.), 1.)
        plt.figure('Exponential pdf')
        plt.plot(x, pdf(x))
        plt.figure('Exponential cdf')
        plt.plot(x, pdf.cdf(x))
        plt.figure('Exponential ppf')
        plt.plot(x, pdf.ppf(x))
        r = pdf.gen_random(1000000)
        plt.figure('Exponential random variate')
        plt.hist(r, bins=100)


if __name__ == '__main__':
    unittest.main(exit=not sys.flags.interactive)
