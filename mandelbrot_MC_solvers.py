import numpy as np
from numpy import ndarray
from scipy.stats.qmc import LatinHypercube

from mandelbrot_MC import mandelbrotIter

from typing import List, Tuple, Dict, Union

class BaseSolver:
    XMIN, XMAX = -2, 1.5
    YMIN, YMAX = -2, 2

    def __init__(self, seed: int=None) -> None:
        self.instantiateRNG(seed)
    
    def instantiateRNG(self, seed: int) -> None:
        """
        Set seef for RNG of solver
        """
        pass

    def samples(self, nSamples: int) -> np.ndarray[float, float]:
        """
        Returns [Nx2] 2D-array of all sampled points (x, y).
        """
        pass
    
    def mandelbrotArea(self, iterations: int, samples: int, power=2, bound=2, scatter=False
                       ) -> Union[float, Tuple[float, List[Tuple[float, float]]]]:
        """
        Implementation of the Monte Carlo integration method for the Mandelbrot set with x boundaries [-2, 1.5] 
        and y boundarys [-2, 2]. Can record a scatter
        plot for all samples that were recorded to be in the mandelbrot set

        Args:
            iterations: Amount of mandelbrot iterations for every value of x+iy
            samples:    Amount of random points to be sampled within the domain
            power:      exponent for mandelbrot iteration (=2)
            bound:      mandelbrot boundary condition (=2)
            scatter:    If true, Return value will be a tuple of the measured area and a list of points
                        for all samples taken in the mandelbrot set.
                        If false, return value will only be the measured area
        
        Returns: The measured mandelbrot area if scatter is False. Else a tuple of recorded area and a list
                of points xy that were recorded to be in the mandelbrot set.
        """
        domainArea = (self.YMAX - self.YMIN) * (self.XMAX - self.XMIN)

        hits = 0
        scatterPoints = []
        samplepoints = self.samples(samples)

        for pointIndex in range(samples):
            x = samplepoints[pointIndex, 0]
            y = samplepoints[pointIndex, 1]

            sampledVal = complex(x, y)
            if not mandelbrotIter(sampledVal, iterations, power, bound):
                hits += 1
                if scatter: scatterPoints.append((x, y))

        area = (hits / samples) * domainArea
        if scatter:
            return area, scatterPoints
        else:
            return area

    def iterate_iterSamples(self, iters: np.ndarray[int], samples: np.ndarray[int]) -> np.ndarray[float, float]:
        """
        Returns Array of estimated areas for different amounts of iterations and samples.

        Along rows, amount of samples remain constant with varying iterations
        Along collumns, amount of iterations remain constant, with variyng amounts of samples 
        
        Args:
            iters:   Array containing all integer amounts of iterations to be tested
            samples: Array containing all integer amounts of samples to be tested
        """
        out = np.zeros((samples.size, iters.size))

        for row, nSamples in enumerate(samples):
            for col, nIter in enumerate(iters):
                area = self.mandelbrotArea(nIter, nSamples,)
                out[row, col] = area

        return out
    
    def iterate_iterSamples_Error(self, iters: np.ndarray[int], samples: np.ndarray[int], sampling:str = 'random'):
        """
        errorI_out = np.zeros((samples.size, iters.size))

        for row, nSamples in enumerate(samples):
            area_iMax = mandelbrotArea(iters[-1], nSamples)
            for col, nIter in enumerate(iters):
                area = mandelbrotArea(nIter, nSamples)
                error = area_iMax - area
                errorI_out[row, col] = error

        return errorI_out
        """
        areas = self.iterate_iterSamples(iters, samples)
        maxAll = areas[-1, -1]

        return abs(maxAll - areas)


class PureRandomSampling(BaseSolver):
    """
    Implementation of the Monte Carlo integration scheme for the Mandelbrot utilizing pure random sampling
    """
    def instantiateRNG(self, seed: int) -> None:
        self.RNG = np.random.default_rng(seed)
    
    def samples(self, nSamples: int) -> ndarray[float, float]:
        return self.RNG.uniform((self.XMIN, self.YMIN), (self.XMAX, self.YMAX), (nSamples, 2))


class LatinHypercubeSampling(BaseSolver):
    """
    Implementation of the Monte Carlo integration scheme for the Mandelbrot utilizing pure random sampling
    """
    def instantiateRNG(self, seed: int) -> None:
        self.RNG = LatinHypercube(2, strength=1, seed=seed)
    
    def samples(self, nSamples: int) -> ndarray[float, float]:
        xDiff = self.XMAX - self.XMIN
        yDiff = self.YMAX - self.YMIN
        samples = self.RNG.random(nSamples)
        samples *= np.asarray([xDiff, yDiff])
        samples += np.asarray([self.XMIN, self.YMIN])

        return samples


class OrthogonalSampling(LatinHypercube):
    """
    Implementation of the Monte Carlo integration scheme for the Mandelbrot utilizing pure random sampling.

    As orthognonal sampling is used, nSamples = p**2 must be used, where p is a prime number and p >= 3
    """
    def instantiateRNG(self, seed: int) -> None:
        self.RNG = LatinHypercube(2, strength=2, seed=seed)