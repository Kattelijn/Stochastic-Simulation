import numpy as np
from numpy import ndarray
from scipy.stats.qmc import LatinHypercube
from time import time
import concurrent.futures

from mandelbrot_MC import mandelbrotIter

from typing import List, Tuple, Dict, Union

HEUR_ITER, HEUR_SAMPLE = 2612, 11881

class BaseSolver:
    def __init__(self, seed: int=None, xDomain=(-2, 2), yDomain=(-2, 2)) -> None:
        self.instantiateRNG(seed)

        self.XMIN, self.XMAX = xDomain
        self.YMIN, self.YMAX = yDomain

    def instantiateRNG(self, seed: int) -> None:
        """
        Set seed for RNG of solver. To be implemented by specific solver.
        """
        raise(NotImplementedError)
    
    def get_state(self) -> int:
        """
        Get current state of the RNG. To be implemented by specific solver
        """
        raise(NotImplementedError)

    def samples(self, nSamples: int) -> np.ndarray[float, float]:
        """
        Returns [Nx2] 2D-array of all sampled points (x, y). To be implemented by specific solver.
        """
        raise(NotImplementedError)
    
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

    def parallelMandelbrotArea(self, iterations: int, samples: int):
        domainArea = (self.YMAX - self.YMIN) * (self.XMAX - self.XMIN)

        hits = 0
        samplepoints = self.samples(samples)
        samplepoints = [complex(samplepoints[i, 0], samplepoints[i,1]) for i in range(samples)]

        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = [executor.submit(mandelbrotIter, point, iterations) for point in samplepoints]

            for future in concurrent.futures.as_completed(futures):
                if not future.result():
                    hits += 1
        
        area = (hits / samples) * domainArea
        return area

    def iterate_iterSamples(self, iters: np.ndarray[int], samples: np.ndarray[int], verbose=False, parallel=False) -> np.ndarray[float, float]:
        """
        Returns Array of estimated areas for different amounts of iterations and samples.

        Along rows, amount of samples remain constant with varying iterations
        Along collumns, amount of iterations remain constant, with variyng amounts of samples 
        
        Args:
            iters:   Array containing all integer amounts of iterations to be tested
            samples: Array containing all integer amounts of samples to be tested
            verbose: If True, function will print the current value nSamples and the time for every outer loop
        """
        out = np.zeros((samples.size, iters.size))

        for row, nSamples in enumerate(samples):
            if verbose: print(f"nSamples: {nSamples}")

            tStartSample = time()
            for col, nIter in enumerate(iters):
                area = None
                if parallel and HEUR_ITER <= nIter and HEUR_SAMPLE <= nSamples: # Parallel calculation only has positive effect for high nIter & nSamples
                    area = self.parallelMandelbrotArea(nIter, nSamples)
                else: 
                    area = self.mandelbrotArea(nIter, nSamples)
                out[row, col] = area
            
            if verbose: print(f"t_nSamples: {time() - tStartSample:.2f}")
        return out
    
    def iterate_iterSamples_Error(self, iters: np.ndarray[int], samples: np.ndarray[int]):
        """
        
        """
        areas = self.iterate_iterSamples(iters, samples)
        maxAll = areas[-1, -1]

        return abs(maxAll - areas)

    def iterSample_std(self, runs: int, iters: np.ndarray[int], samples: np.ndarray[int], trueValParms = (10000, 10000),
                       trueArea: float=None, multiplier=1, verbose=False, parallel = False) -> Tuple[ndarray[float, float], ndarray[float, float], float]:
        """
        Calculates the sample standard deviations for estimated area for different mandelbrot iteration depths 
        and sampling points taken.

        Args:
            runs:         Amount of runs to take sample STD for
            iters:        Array containing all integer amounts of iterations to be tested
            samples:      Array containing all integer amounts of samples to be tested
            trueValParms: Iterations and Samples parameters to estimate the true value from, is ignored if the true area is given
            trueArea:     TrueArea to estimate standard error from, is ignored if None.
        
        Returns: A tuple containing an array of the standard errors w.r.t. the "estimated true value",
            an array of the areas for all runs, and the  "estimated true value".
            
            Along rows, amount of samples remain constant with varying iterations
            Along collumns, amount of iterations remain constant, with variyng amounts of samples 
        """
        results = np.zeros((samples.size, iters.size, runs), float)

        if trueArea is None:
            trueArea = self.mandelbrotArea(*trueValParms) * multiplier

        for run in range(runs):
            tRunStart = time()
            if verbose: print(f"Run {run+1}")

            resultRun = self.iterate_iterSamples(iters, samples, parallel=parallel)
            results[:,:,run] = resultRun * multiplier

            if verbose: print(f"tRun: {time() - tRunStart:.2f}")
            
        
        stds = np.std(results, axis=2, mean=trueArea, ddof=1) # ddof = 1 for unbiased sample standard deviation 

        return stds, results, trueArea


class PureRandomSampling(BaseSolver):
    """
    Implementation of the Monte Carlo integration scheme for the Mandelbrot utilizing pure random sampling
    """
    def instantiateRNG(self, seed: int) -> None:
        self.RNG = np.random.default_rng(seed)
    
    def get_state(self) -> int:
        return self.RNG.bit_generator.state["state"]["state"]
    
    def samples(self, nSamples: int) -> ndarray[float, float]:
        return self.RNG.uniform((self.XMIN, self.YMIN), (self.XMAX, self.YMAX), (nSamples, 2))


class LatinHypercubeSampling(BaseSolver):
    """
    Implementation of the Monte Carlo integration scheme for the Mandelbrot utilizing pure random sampling
    """
    def instantiateRNG(self, seed: int) -> None:
        self.RNG = LatinHypercube(2, strength=1, seed=seed)

    def get_state(self) -> int:
        return self.RNG.rng.bit_generator.state["state"]["state"]
    
    def samples(self, nSamples: int) -> ndarray[float, float]:
        xDiff = self.XMAX - self.XMIN
        yDiff = self.YMAX - self.YMIN
        samples = self.RNG.random(nSamples)
        samples *= np.asarray([xDiff, yDiff])
        samples += np.asarray([self.XMIN, self.YMIN])

        return samples


class OrthogonalSampling(LatinHypercubeSampling):
    """
    Implementation of the Monte Carlo integration scheme for the Mandelbrot utilizing pure random sampling.

    As orthognonal sampling is used, nSamples = p**2 must be used, where p is a prime number and p >= 3
    """
    def instantiateRNG(self, seed: int) -> None:
        self.RNG = LatinHypercube(2, strength=2, seed=seed)