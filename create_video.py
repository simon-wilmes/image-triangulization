import os
import moviepy.video.io.ImageSequenceClip
import magic
IMG_INPUT = 'zpapa-start-bild.png'
IMG_FOLDER = 'img/papa-bild-png'
IMG_TYPE = 'png'
MAX_DIFF_VALUE = 3 # depends on image type (jpg vs png)
NUM_POINTS = 10000

FPS=2

print("Start Generating Image")
image_files = [os.path.join(IMG_FOLDER,img)
            for img in os.listdir(IMG_FOLDER)
            if img.endswith(IMG_TYPE)]
image_files = sorted(image_files)
image_files.remove(IMG_FOLDER  + "/" + IMG_INPUT)
image_files.append(IMG_FOLDER  + "/" + IMG_INPUT)
image_files.append(IMG_FOLDER  + "/" + IMG_INPUT)
clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=FPS)
clip.write_videofile('papa-video-png.mp4')