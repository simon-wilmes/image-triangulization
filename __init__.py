import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
import random
from numpy.random import randint, random
from scipy.spatial import Delaunay
from matplotlib.path import Path
from time import time
from itertools import chain
import tqdm


# Init
# Load image 
IMAGE_INPUT = 'papa-start-bild.png'
IMG_FOLDER = 'img/papa-bild-png'
IMG_TYPE = 'png'

MAX_DIFF_VALUE = 3 # depends on image type (jpg vs png)

NUM_POINTS = 10000

# indices = [6, 10, 15, 21, 28, 36, 45, 55 ....]
indices_save_image = [6]
running_sum = 6
inc = 4
while(running_sum < NUM_POINTS):
    running_sum += inc
    inc += 1
    indices_save_image.append(running_sum)





np.random.seed(seed=1)

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

def iterator_triangle_top(v1, v2, v3, offset = 1):
    invslope1 = (v2[1] - v1[1]) / (v2[0] - v1[0])
    invslope2 = (v3[1] - v1[1]) / (v3[0] - v1[0])
    
    if(invslope2 < invslope1):
        invslope1, invslope2 = invslope2, invslope1 
    
    
    curx1 = v1[1]
    curx2 = v1[1]
    
    for scanline_y in range(v1[0], v2[0] + offset):
        yield (scanline_y, int(np.ceil(curx1)), int(np.floor(curx2) + 1))
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
        yield (scanline_y, int(np.ceil(curx1)), int(np.floor(curx2) + 1))
        curx1 -= invslope1
        curx2 -= invslope2
    

def get_avg_color_triangle(triangle, points, img_sum):
    
    px, py, pz = get_triangle_points_sorted(triangle, points)
    
    avg_color = img_sum[0,0].copy()
    avg_count = 0
    for line, x, y in get_triangle_iterator(px, py, pz):
        avg_color += img_sum[line,y] - img_sum[line,x]
        avg_count += y - x
    return (avg_color / avg_count)


def colorin_triangle(triangle, points, img, color):
    px, py, pz = get_triangle_points_sorted(triangle, points)
    
    for line, x, y in get_triangle_iterator(px, py, pz):
        for j in range(x,y):
            img[line,j] = color




def update_rareness(n):
    # some formula that for n -> infty goes to 1 
    # speed determines how often new points are choosen early relative to later on
    return 1 - 1 / (n / 4)**(1/2) 



def main():
    # IMAGE
    img = mpimg.imread(IMG_FOLDER + "/" + IMAGE_INPUT)
    height, width = len(img), len(img[0])
    
    pixel_size = len(img[0,0])
    # Create running sum array
    print("Calculate Running Sum Array")
    img_sum = np.zeros(shape=(height, width + 1, pixel_size))
    for i in range(height):
        running_sum = img_sum[0,0].copy()
        for j in range(1, width + 1):
            running_sum += img[i,j - 1]
            img_sum[i, j] = running_sum.copy()
    print("Finished Running Sum Array")
    
    points = [(0,0),(0,width - 1),(height - 1,0),(height - 1,width - 1)]
    tri = Delaunay(points)
    start_time = time()
    rareness = update_rareness(len(points))
    
    triangle_col_cal = {}
    
    r = 0
    while(len(points) != NUM_POINTS):
        #print(r, end="\t", flush=True)
        r += 1
        
        random_point = (randint(0, height), randint(0, width))
        # get color
        triangle = tuple(tri.simplices[tri.find_simplex(random_point)])
        # get color of triangle
        
        if(triangle in triangle_col_cal):
            col = triangle_col_cal[triangle]
        else:
            col = get_avg_color_triangle(triangle, points, img_sum)
            triangle_col_cal[triangle] = col
        
        
        difference = np.sum(np.absolute(col - img[random_point]))
        
        if(random() * rareness < (difference / MAX_DIFF_VALUE)**3):
            # if successfull add point
            points.append(random_point)
            # recalculate triangles
            tri = Delaunay(points)
            # update rareness
            rareness = update_rareness(len(points))
            
            print(f"succesfull {len(points)}", end="\t")
            print("new rareness", rareness)
            
            
            # if number of points is in indices_save_image => save_image :)
            if(len(points) in indices_save_image):
                # create copy to draw on
                img_with_points = img.copy()

                triangles = tri.simplices
                color_list = []
                # draw triangles
                for triangle in triangles:
                    color_list.append(get_avg_color_triangle(triangle, points, img_sum))
                    colorin_triangle(triangle, points, img_with_points, color_list[-1])

                # Save image
                print(f"Save Image: {round(time() - start_time,1)}")
                mpimg.imsave(f"{IMG_FOLDER}/tri_image-{len(points):05d}-{r}.{IMG_TYPE}",img_with_points)
            
            

if __name__ == "__main__":
    
    main()
