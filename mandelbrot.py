import numpy as np
import matplotlib.pyplot as plt

def mandelbrot_iteration(xDomain, yDomain, bound=2, power=2, max_iterations=50):
	iterationArray = []

	for y in yDomain:
		row = []
		for x in xDomain:
			c = complex(x,y)
			z = 0
			for iterationNumber in range(max_iterations):
				if(abs(z) >= bound):
					row.append(iterationNumber)
					break
				else: z = z**power + c
			else:
				row.append(0)

		iterationArray.append(row)

	return iterationArray

def plot_mandelbrot(xDomain, yDomain, iterationArray):
	ax = plt.axes()
	graph = ax.pcolormesh(xDomain, yDomain, iterationArray, cmap = 'magma')
	plt.colorbar(graph)
	plt.xlabel("Real-Axis")
	plt.ylabel("Imaginary-Axis")
	plt.show()

n = 1000
a = -2
b = 2
xDomain, yDomain = np.sort(np.random.uniform(a,b,n)), np.sort(np.random.uniform(a,b,n))
bound = 2
power = 2
max_iterations = 50
approximations = np.empty((int(max_iterations/10 - 1), 2))

A_i = mandelbrot_iteration(xDomain, yDomain, max_iterations=max_iterations)
area_A_i = area = (n**2-np.count_nonzero(A_i))/(n**2)*(b-a)**2

for i in range(10, max_iterations, 10):
	iterationArray = mandelbrot_iteration(xDomain, yDomain, max_iterations=i)
	area = (n**2-np.count_nonzero(iterationArray))/(n**2)*(b-a)**2
	approximations[int(i/10)-1] = [i, area - area_A_i]

plt.plot(approximations[:,0], approximations[:,1])
plt.title(rf'Convergence of $A_{{(j,{n})}} - A_{{({max_iterations},{n})}}$ for $j \to {max_iterations}$')
plt.xlabel("Number of iterations j")
plt.ylabel(rf"$A_{{(j,{n})}} - A_{{({max_iterations},{n})}}$")
plt.show()

print('Approximate area of the Mandelbrot set for', n, 'points and', max_iterations, 'iterations:', area)