import numpy as np
import matplotlib.pyplot as plt

xDomain, yDomain = np.linspace(-2,2,500), np.linspace(-2,2,500)
bound = 2
power = 2
max_iterations = 50 

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

print(iterationArray)

ax = plt.axes()
graph = ax.pcolormesh(xDomain, yDomain, iterationArray, cmap = 'magma')
plt.colorbar(graph)
plt.xlabel("Real-Axis")
plt.ylabel("Imaginary-Axis")
plt.show()