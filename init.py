import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
import random
from numpy.random import randint
from scipy.spatial import Delaunay
from matplotlib.path import Path
from time import time
# Init
# Load image 
img = mpimg.imread('family_image.png')

num_points = 10000

ind = range(100,10000,100)

height, width = len(img), len(img[0])

points = [(0,0),(0,width - 1),(height - 1,0),(height - 1,width - 1)]

np.random.seed(seed=0)

pixels_per_run = 90

sweep_square_side = 5 * 2 + 1

random_value = False

r_string = "r" if random_value else "b"

tri = Delaunay(points)
start_time = time()
round = 0
while(len(points) != num_points):
    round += 1
    
    random_point = (randint(0, height), randint(0, width))
    # get color
    
    
    points.append(best_pixel)
    tri = Delaunay(points)
    
    if(round in ind):
        img_with_points = img.copy()

        triangles = tri.simplices
        color_list = []

        for triangle in triangles:
            vx = points[triangle[0]]
            vy = points[triangle[1]]
            vz = points[triangle[2]]
            min_px = min(vx[0],vy[0],vz[0])
            max_px = max(vx[0],vy[0],vz[0])
            min_py = min(vx[1],vy[1],vz[1])
            max_py = max(vx[1],vy[1],vz[1])
            
            avg_col = [0,0,0]
            num_of_v = 0
            
            path = Path([vx,vy,vz])
            
            for i in range(min_px,max_px):
                for j in range(min_py,max_py):
                    if(path.contains_point((i,j))):
                        avg_col += img[i,j]
                        num_of_v += 1
            if(num_of_v == 0):
                avg_col = img[vx[0],vx[1]]
            else:
                avg_col = [x / num_of_v for x in avg_col]
            color_list.append(avg_col)
            
            for i in range(min_px,max_px):
                for j in range(min_py,max_py):
                    if(path.contains_point((i,j))):
                        img_with_points[i,j] = color_list[-1]
        """
        fig = plt.figure("Show two images")

        ax = fig.add_subplot(1,2,1)
        plt.imshow(img)
        ax = fig.add_subplot(1,2,2)
        plt.imshow(img_with_points)
        plt.show()
        """
        # Save image
        print(round, round(time() - start_time,1))
        mpimg.imsave(f"tri_image-{round}-{pixels_per_run}-{r_string}.png",img_with_points)