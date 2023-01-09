import matplotlib.image as mpimg
import numpy as np
from numpy.random import RandomState


from scipy.spatial import Delaunay
from time import time

from itertools import chain
from tqdm import tqdm
import os
import moviepy.video.io.ImageSequenceClip
#import magic
import argparse
from pathlib import Path


parser = argparse.ArgumentParser(prog='Triangulizer',description='A program to "triangualize" an image. And create a video showing the incremental addtions of triangles.')

parser.add_argument("input_img", help="The img to triangulize. Must not contain any transparency.")
parser.add_argument("number_of_points", help="The number of points that will be created.",type=int)
parser.add_argument("--mask_img", help="The Image containing the mask. Where black stands for not increased chance and white for increased chance. (Default: None)")
parser.add_argument("--mask_strength",help="Value between 0 and 1, indicating how much the mask should influence the choosen points. (Default:0.2)",default=0.2,type=float)
parser.add_argument("--store_all_images",help="Whether to store not only the last output image, but all created. (Warning can be a lot of images if many points where choosen).")
parser.add_argument("--distance_points",help="The minimum distance between all points in the image. If choosen too high, program might never finish (Default: 3)",type=float,default=2)
parser.add_argument("--output_folder",help="The name of the output folder in which to store all images and the output video (Default: 'output').",default='output')
parser.add_argument("--video_length",help="The length of the output video in seconds (Default: '5').",default=5,type=float)
parser.add_argument("--video_fps",help="The FPS of the output video (Default: 30).",default=30,type=int)
parser.add_argument("--show_original_img",help="How long the original image is shown at the end of the video in seconds (Default: 3).",default=3,type=float)
parser.add_argument("--random_seed",help="Is used as the seed for the random values. Must be an integer",type=int)
parser.add_argument("--video_speed_up",help="The higher this value the slower the video starts and the quicker it ends. In terms of how many points are added in each frame. Range(0 < x <= 5) (Default: 2)", default=2, type=float)
args = parser.parse_args()









class Triangulizer:
    def __init__(self, args):
        # Parse Args
        self.IMG_INPUT = Path(args.input_img)
        self.USE_MASK = (args.mask_img is not None)
        if(self.USE_MASK):
            self.MASK_INPUT = Path(args.mask_img)
            self.MASK_FACTOR = args.mask_strength
            self.MASK_TRUE_VALUE = 0
        self.IMG_TYPE = self.IMG_INPUT.suffix
        self.NUM_POINTS = args.number_of_points
        self.MIN_DISTANCE_POINTS = args.distance_points
        self.OUTPUT_FOLDER = Path(args.output_folder)
        self.VIDEO_FPS = args.video_fps
        self.VIDEO_SPEED_UP = args.video_speed_up
        self.VIDEO_LENGTH = args.video_length
        self.SHOW_ORIGINAL = args.show_original_img
        self.DIFF_POWER_CONSTANT = 3
        
        self.image_to_max_dtype = {'.png':(1,float),'.jpg':(255, int)}
        
        # Values used for the adaptive algorithm
        self.adaptive_average_num_tries = 50
        self.adaptive_update_value = 0.1
        
        if(args.random_seed is not None):
            self.random = RandomState(args.random_seed)
        else:
            self.random = RandomState(0)
        
        
        
        self.IMAGE_RANGE = 255
        # Create output folder if not exists
        try:
            os.mkdir(self.OUTPUT_FOLDER)
        except:
            # Delete all images in folder if the folder already exists
            for file_name in os.listdir(self.OUTPUT_FOLDER):
                # construct full file path
                file = self.OUTPUT_FOLDER / file_name
                if os.path.isfile(file) and file.suffix in [".png",".jpg", ".mp4"]:
#                    os.remove(file)
                    pass
        
        
        
        # Open image and store only the non-alpha part
        self.img = mpimg.imread(self.IMG_INPUT)[:,:,:3]
        # Normalize image
        self.normalized_img = np.array((self.img - np.min(self.img)) / np.max(self.img) * self.IMAGE_RANGE,dtype=int)
        
        # If mask img was given -> create boolean array mask
        if(self.USE_MASK):
            self.mask_img = mpimg.imread(self.MASK_INPUT)[:,:,:3]
            self.mask = np.array(np.round(np.dot(self.mask_img[...,:3], [0.3333, 0.3333, 0.3334])), dtype=bool)
            
        # Define needed variables
        self.height, self.width = len(self.img), len(self.img[0])
        self.max_diff_value = 3 * self.IMAGE_RANGE
        self.triangles = []
        
        self.img_indices = None
        self.points = None
        self.created_images = None
        self.running_sum = None
        
    def run(self):
        # Create Indices of points of when to store a frame
        self.img_indices = self.calculate_image_indices()
        
        # Create Points and images
        self.points = [(0,0),(0,self.width - 1),(self.height - 1,0),(self.height - 1,self.width - 1)]
        self.points_start_len = 4
        self.created_images = {}
        
        # Calculate Running Sum Image
        self.running_sum = self.calculate_running_sum()



        
        self.create_points_adaptive_color()
        self.create_video(self.created_images)
        
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
        return img_sum

    def calculate_image_indices(self):
        """ Calculates which number of points is to be shown for each frame in the video.

        Returns:
            list[int]: List of number of points. i-th Frame should show image with list[i] points
        """
        num_total_images = self.VIDEO_FPS * self.VIDEO_LENGTH
        
        curve = (lambda x: x**self.VIDEO_SPEED_UP * self.NUM_POINTS)
        indices = [np.floor(curve(image_num / num_total_images)) for image_num in range(int(num_total_images) + 1)]
        return indices
    
    def create_points_adaptive_color(self):
        """
        """
        rareness = 1
        tri = Delaunay(self.points)
        triangle_col_cal = {}
        print("Start Generating Points")
        total_tries = 0
        num_generated = 0
        
        
        for current_point in tqdm(range(self.NUM_POINTS + 1)):
            #print(current_point, len(self.points), )
            num_tries = 0
            point_added = False
            
            # If number of points is in frames to save => create image
            if(len(self.points) - self.points_start_len in self.img_indices):
                self.create_image(triangle_col_cal, num_generated, tri)
            
            
            while(not point_added):
                # Generate a point
                random_point = (self.random.randint(0,self.height),self.random.randint(0,self.width))
                
                for point in self.points:
                    # If the new point is too close to an already existing point. Start Loop new
                    if(self.dist(random_point,point) <= self.MIN_DISTANCE_POINTS):
                        num_generated += 1
                        break 
                else:
                    # If point not too close
                    num_tries += 1
                    total_tries += 1
                    
                    triangle = tuple(tri.simplices[tri.find_simplex(random_point)])
                    
                    triangle_index = tuple(sorted(triangle))
                    # Check if the color of the triangle has already been calculated, otherwise calculate the color
                    if(triangle_index in triangle_col_cal):
                        col = triangle_col_cal[triangle_index]
                    else:
                        col = self.get_avg_color_triangle(triangle)
                        triangle_col_cal[triangle_index] = col
                    
                    # Calculate the difference    
                    diff = np.sum(np.absolute(col - self.normalized_img[random_point]))   
                    
                    if(self.USE_MASK):
                        # If increase mask is used and is true on the random point
                        if(not self.mask[random_point]):
                            # Increase the difference
                            diff = (1 - self.MASK_FACTOR) * diff + self.MASK_FACTOR * self.max_diff_value
                    
                    if(self.random.random() * rareness < (diff / self.max_diff_value)**self.DIFF_POWER_CONSTANT):
                        ## Point was choosen
                        self.points.append(random_point)
                        tri = Delaunay(self.points)
                        # Check if num_tries fits to expected value, and change rareness accordingly

                        if(num_tries < self.adaptive_average_num_tries):
                            rareness *= 1 + self.adaptive_update_value
                        else:
                            rareness *= 1 - self.adaptive_update_value
                            
                        point_added = True
                        
    def create_image(self, triangle_col_cal, num_generated, tri):
        # create copy to draw on
        triangle_img = self.img.copy()

        
        triangles = tri.simplices
        color_list = []
        # draw triangles
        for triangle in triangles:
            col = self.get_avg_color_triangle(triangle)
            triangle_index = tuple(sorted(triangle))
            triangle_col_cal[triangle_index] = col
            
            max_value, type_value = self.image_to_max_dtype[self.IMG_TYPE]
            scaled_col = np.array((col / self.IMAGE_RANGE) * max_value,dtype=type_value)
            color_list.append(scaled_col)
            
            self.colorin_triangle(triangle, triangle_img, scaled_col)
            
        # Create index that starts with as many zeros as needed to display all points
        index = str(len(self.points))
        while(len(index) < len(str(self.NUM_POINTS))):
            index = '0' + index
            
        # Create filename
        filename = self.OUTPUT_FOLDER / (f"image-{index}-{num_generated}" + self.IMG_TYPE)
        self.created_images[len(self.points) - self.points_start_len] = str(filename)
        # Save image
        mpimg.imsave(filename,triangle_img)        


    def dist(self, a, b):
        return ((a[0] - b[0])**2 + (a[1] - b[1])**2)**(1/2)
    
    def get_triangle_points_sorted(self, triangle):
        """ Returns list of three points that make up the triangle which was passed in as an argument. 
        Sorted from top to bottom and left to right.

        Args:
            triangle list[int]: The indices from self.points of the points that make up the triangle.

        Returns:
            list[tuple[int]]: Three points as tuples.
        """
        return sorted([self.points[triangle[0]], self.points[triangle[1]], self.points[triangle[2]]], key=lambda x: x[0] - 1 / (x[1] + 1))

    def create_video(self, created_images): 
        print("Start Generating Video")
        
        pixel_frames = [self.created_images[i] for i in self.img_indices]
        original_frames = [str(self.IMG_INPUT)] * (self.SHOW_ORIGINAL * self.VIDEO_FPS)
        
        total_frames = pixel_frames + original_frames
        
        clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(total_frames, fps=self.VIDEO_FPS)
        clip.write_videofile(str(self.IMG_INPUT.parent / Path(self.IMG_INPUT.stem + ".mp4")))
        
        print("Finish Generating Video")


    def get_triangle_iterator(self, px, py, pz):
        if(px[0] == py[0]):
            iterator = self.iterator_triangle_bottom(px, py, pz, offset=1)
        elif( py[0] == pz[0]):
            iterator = self.iterator_triangle_top(px, py, pz)
        else:
            pt = (py[0], px[1] + (py[0] - px[0]) / (pz[0] - px[0]) * (pz[1] - px[1]))
            if(pt[1] > py[1]):
                py, pt = pt, py
            iterator = chain(self.iterator_triangle_top(px, py, pt), self.iterator_triangle_bottom(py, pt, pz))
        return iterator

    def iterator_triangle_top(self, v1, v2, v3, offset = 1):
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
        
    def iterator_triangle_bottom(self, v1, v2, v3, offset = 0):
        invslope2 = (v2[1] - v3[1]) / (v2[0] - v3[0])
        invslope1 = (v1[1] - v3[1]) / (v1[0] - v3[0])
        
        if(invslope2 > invslope1):
            invslope1, invslope2 = invslope2, invslope1 
        
        
        curx1 = v3[1]
        curx2 = v3[1]
        
        for scanline_y in range(v3[0], v1[0] - offset, -1):
            yield (scanline_y, int(np.ceil(curx1)), int(np.floor(curx2) + 1))
            curx1 -= invslope1
            curx2 -= invslope2
        
    def get_avg_color_triangle(self, triangle):
        
        px, py, pz = self.get_triangle_points_sorted(triangle)
        
        avg_color = np.zeros(3)
        avg_count = 0
        for line, x, y in self.get_triangle_iterator(px, py, pz):
            avg_color += self.running_sum[line,y] - self.running_sum[line,x]
            avg_count += y - x
        if(avg_count == 0):
            print("ERROR COUNT = 0 (Should in theory never happen).")
            avg_count = 1
        return (avg_color / avg_count)
    
    def colorin_triangle(self, triangle, img, color):
        px, py, pz = self.get_triangle_points_sorted(triangle)
        for line, x, y in self.get_triangle_iterator(px, py, pz):
            for j in range(x,y):
                img[line,j] = color
    


if __name__ == "__main__":
    
    triangulizer = Triangulizer(args)
    triangulizer.run()
