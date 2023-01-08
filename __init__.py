import matplotlib.image as mpimg
import numpy as np
from numpy.random import RandomState


from scipy.spatial import Delaunay
from time import time

from itertools import chain
from tqdm import tqdm
import os
import moviepy.video.io.ImageSequenceClip
import magic
import argparse
from pathlib import Path


parser = argparse.ArgumentParser(prog='Triangulizer',description='A program to "triangualize" an image. And create a video showing the incremental addtions of triangles.')

parser.add_argument("input_img", help="The img to triangulize. Must not contain any transparency.")
parser.add_argument("number_of_points", help="The number of points that will be created.",type=int)
parser.add_argument("--mask_img", help="The Image containing the mask. Where black stands for not increased chance and white for increased chance. (Default: None)")
parser.add_argument("--mask_strength",help="Value between 0 and 1, indicating how much the mask should influence the choosen points. (Default:0.2)",default=0.2,type=float)
parser.add_argument("--store_all_images",help="Whether to store not only the last output image, but all created. (Warning can be a lot of images if many points where choosen).")
parser.add_argument("--distance_points",help="The minimum distance between all points in the image. If choosen too high, program might never finish (Default: 3)",type=float,default=3)
parser.add_argument("--output_folder",help="The name of the output folder in which to store all images and the output video (Default: 'output').",default='output')
parser.add_argument("--video_length",help="The length of the output video in seconds (Default: '5').",default=5,type=float)
parser.add_argument("--video-fps",help="The FPS of the output video (Default: 30).",default=30,type=int)
parser.add_argument("--show_original_img",help="How long the original image is shown at the end of the video in seconds (Default: 3).",default=3,type=float)
parser.add_argument("--random_seed",help="Is used as the seed for the random values.",type=float)

args = parser.parse_args()







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
    if(avg_count == 0):
        print("ERROR COUNT = 0" , triangle, points, img_sum)
        avg_count = 1
    return (avg_color / avg_count)


def colorin_triangle(triangle, points, img, color):
    px, py, pz = get_triangle_points_sorted(triangle, points)
    
    for line, x, y in get_triangle_iterator(px, py, pz):
        for j in range(x,y):
            img[line,j] = color


def calculate_images():
    pass

def create_video():
    pass
class Triangulizer:
    def __init__(self, args):
        # Parse Args
        self.IMG_INPUT = Path(args.input_img)
        self.MASK_INPUT = Path(args.mask_img)
        self.USE_MASK = (args.mask_img is not None)
        self.MASK_FACTOR = args.mask_strength
        self.MASK_TRUE_VALUE = 0
        self.IMG_FOLDER = args.output_folder
        self.IMG_TYPE = self.IMG_INPUT.suffix
        self.NUM_POINTS = args.number_of_points
        self.DISTANCE_BETWEEN_POINTS = args.distance_points
        self.OUTPUT_FOLDER = Path(args.output_folder)
        self.VIDEO_FPS = args.video_fps
        self.VIDEO_SPEED_UP = 2
        self.VIDEO_LENGTH = args.video_length
        self.SHOW_ORIGINAL = args.show_original_img
        
        if(args.random_seed is not None):
            self.random= RandomState(args.random_seed)

        
        
        self.IMAGE_RANGE = 255
        # Create output folder if not exists
        try:
            os.mkdir(self.FOLDER)
        except:
            pass
        
        # Open image and store only the non-alpha part
        self.img = mpimg.imread(self.IMG_INPUT)[:,:,:3]
        # Normalize image
        self.normalized_img = np.array((self.img - np.min(self.img)) / np.max(self.img) * self.IMAGE_RANGE,dtype=int)
        
        # If mask img was given -> create boolean array mask
        if(self.USE_MASK):
            self.mask_img = mpimg.imread(self.MASK_INPUT)
            self.mask = np.array(np.round(np.dot(self.mask_img[...,:3], [0.3333, 0.3333, 0.3334])), dtype=bool)
            
        
        self.height, self.width = len(self.img), len(self.img[0])
        self.max_diff_value = 3 * self.IMAGE_RANGE
        
        
        
        
        
        # Create Indices of points of when to store a frame
        self.img_indices = self.calculate_image_indices()
        
        # Create Points and images
        self.points = [(0,0),(0,self.width - 1),(self.height - 1,0),(self.height - 1,self.width - 1)]
        self.created_images = []
        
        # Calculate Running Sum Image
        self.running_sum = self.calculate_running_sum()



        
        self.create_points_adaptive_color()
        #self.create_video(self.created_images)
        
        print("Finished")
    def calculate_running_sum(self):
        """ Calculates an array which stores in the i-th entry the sum of that row from the first
            to the i-th element. To calculate the sum from the i-th to the j-th Element use array[j] - array[i]
        Returns:
            list[int]: The array that contains the running sum. It has dimension height x (width + 1)
        """
        print("Calculate Running Sum Array")
        img_sum = np.zeros(shape=(self.height, self.width + 1, 3))
        for i in tqdm(range(self.height)):
            running_sum = [0] * 3
            for j in range(1, self.width + 1):
                running_sum += self.normalized_img[i,j - 1]
                img_sum[i, j] = running_sum.copy()
        print("Finished Running Sum Array")
        return running_sum

    def calculate_image_indices(self):
        """ Calculates which number of points is to be shown for each Frame in the Video. All indices are

        Returns:
            list[int]: List of number of points. i-th Frame should show image with list[i] points
        """
        num_total_images = self.VIDEO_FPS * self.VIDEO_LENGTH
        
        curve = (lambda x: x**self.VIDEO_SPEED_UP * self.NUM_POINTS)
        indices = [np.floor(curve(image_num / num_total_images)) for image_num in range(int(num_total_images) + 1)]
        return indices
    
    def create_points_adaptive_color(self):
        
        pass
        
def main():

    tri = Delaunay(points)
    start_time = time()
    rareness = update_rareness(len(points))
    
    triangle_col_cal = {}
    print("Start Generating Points")
    with tqdm(total=NUM_POINTS) as pbar:
        r = 0
        while(len(points) != NUM_POINTS):
            #print(r, end="\t", flush=True)
            
            
            random_point = (randint(0, height), randint(0, width))
            
            #check if any point is closer than min_distance
            for point in points:
                if((point[0] - random_point[0])**2 + (point[1] - random_point[1])**2 <= DISTANCE_BETWEEN_POINTS**2):
                    break
            else:
                # get color
                r += 1
                triangle = tuple(tri.simplices[tri.find_simplex(random_point)])
                # get color of triangle
                
                if(triangle in triangle_col_cal):
                    col = triangle_col_cal[triangle]
                else:
                    col = get_avg_color_triangle(triangle, points, img_sum)
                    triangle_col_cal[triangle] = col
                
                
                difference = np.sum(np.absolute(col - img[random_point]))
                
                tmp_mask_factor = MASK_FACTOR if (mask_img[random_point[0], random_point[1]] == MASK_TRUE_VALUE).all() else 1
               
                if(random() * rareness * tmp_mask_factor < (difference / MAX_DIFF_VALUE)**POWER_CONSTANT):
                    # if successfull add point
                    points.append(random_point)
                    # recalculate triangles
                    tri = Delaunay(points)
                    # update rareness
                    rareness = update_rareness(len(points))
                    
                    pbar.update(1)
                    
                    
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
                        #print(f"Save Image: {round(time() - start_time,1)}")
                        mpimg.imsave(f"{IMG_FOLDER}/tri_image-{len(points):05d}-{r}.{IMG_TYPE}",img_with_points)
    print("Finished Generating Points")
    print("Start Generating Video")
    image_files = [os.path.join(IMG_FOLDER,img)
               for img in os.listdir(IMG_FOLDER)
               if img.endswith(IMG_TYPE)]
    image_files = sorted(image_files)
    try:
        image_files.remove(IMG_FOLDER  + "/" + IMG_INPUT)
    except:
        pass
    try:
        image_files.remove(IMG_FOLDER + "/" + MASK_INPUT)
    except:
        pass
    image_files += [IMG_FOLDER  + "/" + IMG_INPUT for _ in range(NUMBER_ORIGINAL_FRAME_SECONDS * FPS)]
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=FPS)
    clip.write_videofile(IMG_INPUT.rsplit('.',1)[0] + ".mp4")
    
            

if __name__ == "__main__":
    
    t = Triangulizer(args)
    print("Finished")
