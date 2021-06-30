from gym.spaces import Box
import numpy as np
import torch as T
from gym.spaces import Dict
import cv2

IM_WIDTH = 64
IM_HEIGHT = 64
N_CHAN = 3


class ActionSpace():
    def __init__(self):
        self.action_spaces = Box(low=np.array([-1,-3]),high=np.array([1,3]),shape=(2,),dtype=np.float32) # Steer,Acc

class CameraOnlyObservation():
    def __init__(self):
        self.observation_space = Box(low=0,high=255,dtype=np.uint8,shape=(N_CHAN,IM_HEIGHT,IM_WIDTH))

class ObservationSpace():
    def __init__(self):
        observations_list = {
            'camera':Box(low=0,high=255,dtype=np.uint8,shape=(N_CHAN,IM_HEIGHT,IM_WIDTH)),
            'state': Box(low = np.array([-2,-1,-5,0]), high= np.array([2,1,30,1]), shape=(4,),dtype=np.float32) # steer,acc,speed,offroad,other_lane
        }
        self.observation_space = Dict(observations_list)

"""if __name__ == "__main__":
   a = ObservationSpace()
   obs = a.observation_space.sample()
   cam,state = obs['camera'],obs['state']
   cam = T.tensor([cam],dtype=T.float)
   print(cam.shape)"""



