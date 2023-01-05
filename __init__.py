import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
import random
from numpy.random import randint, random
from scipy.spatial import Delaunay
from matplotlib.path import Path
from time import time
from itertools import chain

# Init
# Load image 
IMAGE_INPUT = 'papa-start-bild.jpg'
IMG_FOLDER = 'img/papa-bild'
IMG_TYPE = 'png'



num_points = 10000

indices_save_image = [6]
s = 6
k = 4
while(s < num_points):
    s += k
    k += 1
    indices_save_image.append(s)





np.random.seed(seed=0)

def get_triangle_points_sorted(triangle, points):
    return sorted([points[triangle[0]], points[triangle[1]], points[triangle[2]]], key=lambda x: x[0] - 1 / (x[1] + 1))

def get_triangle_iterator(px, py, pz):
    if(px[0] == py[0]):
        iterator = iterator_triangle_bottom(px, py, pz, offset=1)
    elif( py[0] == pz[0]):
        iterator = iterator_triangle_top(px, py, pz)
    else:
        pt = (py[0], px[1] + (py[0] - px[0]) / (pz[0] - px[0]) * (pz[1] - px[1]))
        
        iterator = chain(iterator_triangle_top(px, py, pt), iterator_triangle_bottom(py, pt, pz))
    return iterator


def get_avg_color_triangle(triangle, points, img_sum):
    
    px, py, pz = get_triangle_points_sorted(triangle, points)
    avg_color = img_sum[0,0].copy()
    avg_count = 0
    for line, x, y in get_triangle_iterator(px, py, pz):
        avg_color += img_sum[line,y] - img_sum[line,x]
        avg_count += y - x
    return (avg_color / avg_count)

def iterator_triangle_top(v1, v2, v3, offset = 1):
    invslope1 = (v2[1] - v1[1]) / (v2[0] - v1[0])
    invslope2 = (v3[1] - v1[1]) / (v3[0] - v1[0])
    
    if(invslope2 < invslope1):
        invslope1, invslope2 = invslope2, invslope1 
    
    
    curx1 = v1[1]
    curx2 = v1[1]
    
    for scanline_y in range(v1[0], v2[0] + offset):
        yield (scanline_y, np.ceil(curx1), np.floor(curx2) + 1)
        curx1 += invslope1
        curx2 += invslope2
    
def iterator_triangle_bottom(v1, v2, v3, offset = 0):
    invslope2 = (v2[1] - v3[1]) / (v2[0] - v3[0])
    invslope1 = (v1[1] - v3[1]) / (v1[0] - v3[0])
    
    if(invslope2 > invslope1):
        invslope1, invslope2 = invslope2, invslope1 
    
    
    curx1 = v3[1]
    curx2 = v3[1]
    
    for scanline_y in range(v3[0], v1[0] + offset, -1):
        yield (scanline_y, np.ceil(curx1), np.floor(curx2) + 1)
        curx1 -= invslope1
        curx2 -= invslope2
    
def colorin_triangle(triangle, points, img, color):
    vx = points[triangle[0]]
    vy = points[triangle[1]]
    vz = points[triangle[2]]





def rare(n):
    return 5 - 9 * 1 / n**(1/2)


def main():
    # IMAGE
    img = mpimg.imread(IMG_FOLDER + "/" + IMAGE_INPUT)
    height, width = len(img), len(img[0])
    
    pixel_size = len(img[0,0])
    # Create running sum array
    
    img_sum = np.zeros(shape=(height, width + 1, pixel_size))
    for i in range(height):
        running_sum = img_sum[0,0].copy()
        for j in range(1, width + 1):
            running_sum += img[i,j]
            img_sum[i, i] = running_sum.copy()
    
    
    points = [(0,0),(0,width - 1),(height - 1,0),(height - 1,width - 1)]
    tri = Delaunay(points)
    start_time = time()
    rareness = 1
    
    
    r = 0
    while(len(points) != num_points):
        print(r, end="\t", flush=True)
        r += 1
        
        random_point = (randint(0, height), randint(0, width))
        # get color
        triangle = tri.simplices[tri.find_simplex(random_point)]
        # get color of triangle
        
        col = get_avg_color_triangle(triangle, points, img)
        
        difference = np.sum(np.absolute(col - img[random_point]))
        
        if(random() * rareness < difference):
            points.append(random_point)
            tri = Delaunay(points)
            print(f"succesfull {len(points)}", end="\t")
            rareness = rare(len(points))
            print("new rareness", rareness)
            if(len(points) in indices_save_image):
                img_with_points = img.copy()

                triangles = tri.simplices
                color_list = []

                for triangle in triangles:
                    color_list.append(get_avg_color_triangle(triangle, points, img))
                    colorin_triangle(triangle, points, img_with_points, color_list[-1])

                # Save image
                print("saveimage", r, round(time() - start_time,1))
                mpimg.imsave(f"{IMG_FOLDER}/tri_image-{len(points):05d}-{r}.{IMG_TYPE}",img_with_points)
            
            

if __name__ == "__main__":
    print(get_triangle_points_sorted([0,1,2],[(0,3),(0,2),(1,0)]))
    for i in iterator_triangle_top((0,3),(3,0),(3,4.5)):
        print(i)
    for i in iterator_triangle_bottom((0,1),(0,20),(10,10)):
        print(i)
    pass
    #main()
