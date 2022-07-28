import scipy
from scipy.spatial import Voronoi, voronoi_plot_2d
import numpy as np
import matplotlib.pyplot as plt

points = []
N = 100
MAX_ITERATION = 6
rho = 0.2

# TODO Clip the point dimmension to [0,1] when computing centroid

for i in range(N):
    points.append(np.random.random(2))
points = np.array(points)


for it in range(MAX_ITERATION):

    vor = Voronoi(points, qhull_options = "Qc")
    
    centroids = []

    for k in range(N):
        center = np.zeros(2)

        for vtx in vor.regions[vor.point_region[k]]:
            center += np.clip(vor.vertices[vtx], 0,1)
        center = center/ len(vor.regions[vor.point_region[k]])
        centroids.append(center)

    ## move toward centroids
    for k in range(N):
        points[k] = (1-rho)*points[k] + rho*centroids[k]


fig = voronoi_plot_2d(vor)
plt.show()