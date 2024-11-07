import numpy as np
from numpy import ndarray

from typing import List, Tuple, Dict

XMIN, XMAX = -2, 1.5
YMIN, YMAX = -2, 2

generator: np.random.Generator = None


def instantiateRNG(seed = None):
    global generator
    generator = np.random.default_rng(seed)


def mandelbrotIter(val: np.complexfloating, nIter: int, power=2, bound=2):
    """
    Performs the mandelbrot iteration scheme for a single complex value

    Args:
        nIter:   Amount of mandelbrot iterations for every value of x+iy
        power:   exponent for mandelbrot iteration
        bound:   mandelbrot boundary condition
    """
    z = 0
    for iter in range(nIter):
        z = np.power(z, power) + val

        if z >= bound:
            return iter
    
    return 0


def mandelbrotDomain(xDomain: ndarray[float], yDomain: ndarray[float], nIter: int, power=2, bound=2, 
                     graphicsBuffer: ndarray = None):
    """
    Performs the mandelbrot iteration scheme over the x and y domain and returns the array containing
    the iterations required to surpass the boundary conditions or 0 if the point lies in the mandelbrot set.

    Args:
        xDomain: linspace of values in the x Domain
        yDomain: linspace of values in the y domain
        nIter:   Amount of mandelbrot iterations for every value of x+iy
        power:   exponent for mandelbrot iteration (=2)
        bound:   mandelbrot boundary condition (=2)
        graphicsBuffer: buffer output array for memory efficient animation
    
    
    returns:
        the array containing the iterations required to surpass the boundary conditions or 0 if the 
        point lies in the mandelbrot set.
 
    """
    if graphicsBuffer is None:
        graphicsBuffer = np.zeros((yDomain.shape[0], xDomain.shape[0]), int)

    for col, xVal in enumerate(xDomain):
        for row, yVal in enumerate(yDomain):
            inVal = complex(xVal, yVal)
            outVal = mandelbrotIter(inVal, nIter, power, bound)

            graphicsBuffer[row, col] = outVal

    return graphicsBuffer


def mandelbrotArea(iterations: int, samples: int, power=2, bound=2, scatter=False):
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
    domainArea = (YMAX - YMIN) * (XMAX - XMIN)

    hits = 0
    scatterPoints = []

    for sample in range(samples):
        x = generator.uniform(XMIN, XMAX)
        y = generator.uniform(YMIN, YMAX)
        sampledVal = complex(x, y)
        if not mandelbrotIter(sampledVal, iterations, power, bound):
            hits += 1
            if scatter: scatterPoints.append((x, y))

    area = (hits / samples) * domainArea
    if scatter:
        return area, scatterPoints
    else:
        return area


def iterate_iterSamples_Error(iters: np.ndarray[int], samples: np.ndarray[int]):
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
    areas = iterate_iterSamples(iters, samples)
    maxAll = areas[-1, -1]

    return abs(maxAll - areas)

    
    

def iterate_iterSamples(iters: np.ndarray[int], samples: np.ndarray[int]):

    out = np.zeros((samples.size, iters.size))

    for row, nSamples in enumerate(samples):
        for col, nIter in enumerate(iters):
            area = mandelbrotArea(nIter, nSamples)
            out[row, col] = area

    return out

instantiateRNG()