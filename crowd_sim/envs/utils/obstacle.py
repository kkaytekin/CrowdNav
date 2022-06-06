from typing import List
from crowd_sim.envs.utils.agent import Agent
from pyparsing import string
import numpy as np
from crowd_sim.envs.utils.state import ObservableState, FullState

class Obstacle(object):
    def __init__(self, vertices: List = None, type: string = "Rectangle"):
        self.vertices = vertices
        self.type = type
        self.px, self.py = self.get_center()
        self.l = self.get_length()
        self.w = self.get_width()
        """
        ## In order to meet with the original input
        ## the other agent would see an obstacle as a circle
        """
        self.radius = self.get_equivalent_radius()

    def get_rvo2_obstacle_state(self):
        return self.type, self.vertices

    def get_center(self):
        cx = 0
        cy = 0
        for (x, y) in self.vertices:
            cx += x
            cy += y
        
        cx = cx / 4
        cy = cy / 4

        return cx, cy

    def get_length(self):
        x1, y1 = self.vertices[0]
        x2, y2 = self.vertices[1]

        return np.abs(x1 - x2)

    def get_width(self):
        x1, y1 = self.vertices[0]
        x4, y4 = self.vertices[3]

        return np.abs(y1 - y4)
    
    def get_equivalent_radius(self):
        return np.sqrt((self.w / 2) ** 2 + (self.l / 2) ** 2) 

    def get_observable_state(self):
        return ObservableState(self.px, self.py, 0, 0, self.radius)


    




