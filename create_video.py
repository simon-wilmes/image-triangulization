import os
import moviepy.video.io.ImageSequenceClip
import magic
image_folder='img/papa-bild-png'
fps=2

image_files = [os.path.join(image_folder,img)
               for img in os.listdir(image_folder)
               if img.endswith(".png")]
image_files = sorted(image_files)

for i in image_files:
    print(magic.from_file(i))
clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
clip.write_videofile('papa-video-png.mp4')