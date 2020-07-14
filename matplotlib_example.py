import numpy as np
import matplotlib.pyplot as plt

from numpy.random import uniform

N = 5000
radius = 1.
area = (2*radius)**2

pts = uniform(-1,1,(N, 2))
dist = np.linalg.norm(pts, axis = 1)
in_circle = dist <= 1

pts_in_circle = np.count_nonzero(in_circle)
pi = 4 * (pts_in_circle / N)

plt.scatter(pts[in_circle,0], pts[in_circle,1], marker = ',', edgecolor = 'k', s = 1)
plt.scatter(pts[~in_circle,0], pts[~in_circle,1], marker=',', edgecolor='r', s=1)
plt.axis('equal')

print('mean pi(N={})= {:.4f}'.format(N, pi))
print('err  pi(N={})= {:.4f}'.format(N, np.pi-pi))
plt.show()