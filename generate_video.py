"""
This program is a helper module
for saving PyGame animations

Author: Brendan Halliday
Queen's University
Last edited: april 24th 2024
"""

import pygame
import os
import sys


class Movie_Maker:
 
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
    
    def png(self,screen):
        self.cnt+=1
        fullpath = self.path + "\\"+self.name + "%08d.png"%self.cnt
        pygame.image.save(screen,fullpath)
 
    def mp4(self):
        os.system("ffmpeg -r 60 -i pngs\\capture%08d.png -vcodec mpeg4 -q:v 0 -y fly_mov.mp4")
 
 
if __name__  == '__main__':
    video = Movie_Maker((1280,720))
    video.mp4()