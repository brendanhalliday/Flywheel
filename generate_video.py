import pygame
import os
import sys


class Video:
 
    def __init__(self, size):
        directory_name = os.path.dirname(__file__) # find directory of current running python file
        os.chdir(directory_name) # change the working directory to the directory that includes the running file
        
        self.path = directory_name + "\\pngs"
        self.name = "capture"
        self.cnt = 0
 
        # Ensure we have somewhere for the frames
        try:
            os.makedirs(self.path)
        except OSError:
            pass
    
    def make_png(self,screen):
        self.cnt+=1
        fullpath = self.path + "\\"+self.name + "%08d.png"%self.cnt
        pygame.image.save(screen,fullpath)
 
    def make_mp4(self):
        os.system("ffmpeg -r 60 -i pngs\\capture%08d.png -vcodec mpeg4 -q:v 0 -y fly_mov.mp4")
 
 
if __name__  == '__main__':
    video = Video((1280,720))
    video.make_mp4()