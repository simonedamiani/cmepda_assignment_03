"""Third assignment of CMEPDA Course, 2022/2023"""

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline


class ProbabilityDensityFunction(InterpolatedUnivariateSpline):
    """Class describing a probability density function"""

    def __init__(self, x, y, k=3):
        """Constructor of the probability density function"""
        # Create and normalize (if it's not) the pdf
        normalization = InterpolatedUnivariateSpline(x, y, k=k).integral(x[0], x[-1])
        y /= normalization
        InterpolatedUnivariateSpline.__init__(self, x, y, k=k)

        # Create the cdf
        y_cdf = np.array([self.integral(x[0], x_cdf) for x_cdf in x])
        self.cdf = InterpolatedUnivariateSpline(x, y_cdf, k=k)

        # Create the ppf
        x_ppf, index_ppf = np.unique(y_cdf, return_index=True)
        y_ppf = x[index_ppf]
        self.ppf = InterpolatedUnivariateSpline(x_ppf, y_ppf, k=k)

    def probability(self, x_1, x_2):
        """Return the probability evaluated using the CDF distribution"""
        return self.cdf(x_2) - self.cdf(x_1)

    def gen_random(self, size=1000):
        """Return an array of random value generated according to the chosen PDF"""
        return self.ppf(np.random.uniform(size=size))
