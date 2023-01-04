import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
import random
from numpy.random import randint, random
from scipy.spatial import Delaunay
from matplotlib.path import Path
from time import time
# Init
# Load image 
IMAGE_INPUT = 'family_image.png'
IMG_FOLDER = 'img'


img = mpimg.imread(IMG_FOLDER + "/" + IMAGE_INPUT)

num_points = 10000

indices_save_image = range(100,10000,100)

height, width = len(img), len(img[0])



np.random.seed(seed=0)


RARENESS = 20

def get_avg_color_triangle(triangle, points, img):
    vx = points[triangle[0]]
    vy = points[triangle[1]]
    vz = points[triangle[2]]
    min_px = min(vx[0],vy[0],vz[0])
    max_px = max(vx[0],vy[0],vz[0])
    min_py = min(vx[1],vy[1],vz[1])
    max_py = max(vx[1],vy[1],vz[1])
    
    avg_col = [0] * len(img[0,0])
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
    return avg_col

def colorin_triangle(triangle, points, img, color):
    vx = points[triangle[0]]
    vy = points[triangle[1]]
    vz = points[triangle[2]]
    min_px = min(vx[0],vy[0],vz[0])
    max_px = max(vx[0],vy[0],vz[0])
    min_py = min(vx[1],vy[1],vz[1])
    max_py = max(vx[1],vy[1],vz[1])
    path = Path([vx,vy,vz])
    for i in range(min_px,max_px):
        for j in range(min_py,max_py):
            if(path.contains_point((i,j))):
                img[i,j] = color

def main():
    round = 0
    points = [(0,0),(0,width - 1),(height - 1,0),(height - 1,width - 1)]
    tri = Delaunay(points)
    start_time = time()
    
    while(len(points) != num_points):
        round += 1
        
        random_point = (randint(0, height), randint(0, width))
        # get color
        triangle = tri.find_simplex(random_point)
        # get color of triangle
        
        col = get_avg_color_triangle(triangle, points, img)
        
        difference = np.sum(np.absolute(col - img[random_point]))
        
        if(random() * RARENESS < difference):
            points.append(random_point)
            tri = Delaunay(points)
        
            if(len(points) in indices_save_image):
                img_with_points = img.copy()

                triangles = tri.simplices
                color_list = []

                for triangle in triangles:
                    color_list.append(get_avg_color_triangle(triangle, points, img))
                    colorin_triangle(triangle, points, img, color_list[-1])

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
                mpimg.imsave(f"{IMG_FOLDER}/tri_image-{round}-{pixels_per_run}.png",img_with_points)
            
            

if __name__ == "__main__":
    main()