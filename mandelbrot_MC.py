import numpy as np
from numpy import ndarray
from math import isqrt

from typing import List, Tuple, Dict
from scipy.stats.qmc import LatinHypercube

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
        xDomain: linspace of values in the x domain
        yDomain: linspace of values in the y domain
        sampling: sampling technique to generate the domain
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

def prime_sieve(n: int):
    """
    Perform the sieve of Eratosthenes up to a given bound n.
    Returns a list of primes and the number of cross-out operations performed.
    """
    # Initialize list of numbers from 2 to n
    numbers = list(range(2, n + 1))
    
    cross_outs = 0  # To count cross-out operations
    j = 0  # Start index for sieve
    
    # Apply sieve
    while numbers[j] <= isqrt(n):
        for i in range(j + numbers[j], len(numbers), numbers[j]):
            if numbers[i] != 0:
                numbers[i] = 0  # Mark as non-prime
                cross_outs += 1
        
        # Move j to the next non-zero (prime) element
        j += 1
        while j < len(numbers) and numbers[j] == 0:
            j += 1
    
    # Collect only the primes (non-zero values)
    primes = [num for num in numbers if num != 0]
    return primes
