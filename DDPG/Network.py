from observation_action_space import *
import os
import torch as T

import torch.nn as nn
import  torch.nn.functional as F
import numpy as np
from gym import spaces
import torch.optim as optim
import logging


fc_num = 32 * 29 * 29# fc_layers

class ActorNetwork(nn.Module):
    def __init__(self,alpha,measures_state_dims,fc1,fc2,actions,chkpt_name,chkpt_dir="models/ddpg",n_channels=N_CHAN):
        super().__init__()

        self.chkt_name = chkpt_name
        self.chkt_dir = chkpt_dir
        self.chkt_file = os.path.join(self.chkt_dir,self.chkt_name+"_ac_ddpg")
        self.n_channels = n_channels
        self.fc1 = fc1
        self.fc2 = fc2
        self.n_actions = actions
        self.measure_state_dims = measures_state_dims

        self.cnn_layers = nn.Sequential(
            nn.Conv2d(self.n_channels, 16, 5, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 5, 2),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
    
        self.fc_layers = nn.Sequential(
            nn.Linear(fc_num,self.fc1),
            nn.LayerNorm(self.fc1),
            nn.ReLU(),
            
            nn.Linear(self.fc1,self.fc2),
            nn.LayerNorm(self.fc2),
            nn.ReLU()
        )
        self.measures_layers = nn.Sequential(
            nn.Linear(*self.measure_state_dims,self.fc1),
            nn.LayerNorm(self.fc1),
            nn.ReLU(),

            nn.Linear(self.fc1,self.fc2),
            nn.LayerNorm(self.fc2),
            nn.ReLU()

        )
        self.mu_layer = nn.Sequential(
            nn.Linear(self.fc2 * 2 ,self.n_actions),
            nn.Tanh()
        )

        f1 = 1. / np.sqrt(self.fc_layers[0].weight.data.size()[0])
        self.fc_layers[0].weight.data.uniform_(-f1, f1)
        self.fc_layers[0].bias.data.uniform_(-f1, f1)

        f2 = 1. / np.sqrt(self.fc_layers[1].weight.data.size()[0])
        self.fc_layers[1].weight.data.uniform_(-f2, f2)
        self.fc_layers[1].bias.data.uniform_(-f2, f2)

        f3 = 1. / np.sqrt(self.measures_layers[0].weight.data.size()[0])
        self.measures_layers[0].weight.data.uniform_(-f3, f3)
        self.measures_layers[0].bias.data.uniform_(-f3, f3)

        f4 = 1. / np.sqrt(self.measures_layers[1].weight.data.size()[0])
        self.measures_layers[1].weight.data.uniform_(-f4, f4)
        self.measures_layers[1].bias.data.uniform_(-f4, f4)

        f5 = 0.003
        self.mu_layer[0].weight.data.uniform_(-f5,f5)
        self.mu_layer[0].bias.data.uniform_(-f5,f5)

        self.optim = optim.Adam(self.parameters(),lr=alpha)
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self,img_state,measure_state):
        """
        compute for Image
        compute for measures
        concate and pass to fc layers
        """
        for layer in self.cnn_layers:
            img_state= layer(img_state)
            #print(img_state.view(img_state.size(0),-1).shape)

        img_state = img_state.view(img_state.size(0),-1)
        #print("-",img_state.view(-1,fc_num).shape)

        for layer in self.fc_layers:
            img_state  = layer(img_state)

        for layer in self.measures_layers:
            measure_state = layer(measure_state)

        #TODO remove squzee testing param for img_state
        img_measure_state =T.cat([img_state,measure_state],dim=-1)
        #print(img_measure_state.shape)

        for layer in self.mu_layer:
            img_measure_state = layer(img_measure_state)
        # print(measure_state.size())
        # print(T.cat((img_state.squeeze(),measure_state),dim=0).shape)
        return img_measure_state

    def save_model_checkpoint(self):
        logging.info("Saving the model")
        T.save(self.state_dict(),self.chkt_file)

    def load_model_checkpoint(self):
        logging.info("Loading the model")
        self.load_state_dict(T.load(self.chkt_file))

    def save_best_checkpoint(self):
        logging.info("Saving the best model")
        c_f = os.path.join(self.chkt_dir,self.chkt_name+"_ac_best")
        T.save(self.state_dict(),c_f)


class CriticNetwork(nn.Module):
    def __init__(self, beta, measures_state_dims, fc1, fc2, actions, chkpt_name, chkpt_dir="models/ddpg",n_channels=N_CHAN):
        super().__init__()

        self.chkt_name = chkpt_name
        self.chkt_dir = chkpt_dir
        self.chkt_file = os.path.join(self.chkt_dir, self.chkt_name + "_cn_ddpg")
        self.n_channels = n_channels
        self.fc1 = fc1
        self.fc2 = fc2
        self.n_actions = actions
        self.measure_state_dims = measures_state_dims

        self.cnn_layers = nn.Sequential(
            nn.Conv2d(N_CHAN, 16, 5, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 5, 2),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(fc_num, self.fc1),
            nn.LayerNorm(self.fc1),
            nn.ReLU(),

            nn.Linear(self.fc1, self.fc2),
            nn.LayerNorm(self.fc2),
            nn.ReLU()
        )
        self.measures_layers = nn.Sequential(
            nn.Linear(*self.measure_state_dims, self.fc1),
            nn.LayerNorm(self.fc1),
            nn.ReLU(),

            nn.Linear(self.fc1, self.fc2),
            nn.LayerNorm(self.fc2),
            nn.ReLU()

        )
        #weights for convolution FC layers and measurements linear layers
        f1 = 1. / np.sqrt(self.fc_layers[0].weight.data.size()[0])
        self.fc_layers[0].weight.data.uniform_(-f1, f1)
        self.fc_layers[0].bias.data.uniform_(-f1, f1)

        f2 = 1. / np.sqrt(self.fc_layers[1].weight.data.size()[0])
        self.fc_layers[1].weight.data.uniform_(-f2, f2)
        self.fc_layers[1].bias.data.uniform_(-f2, f2)

        f3 = 1. / np.sqrt(self.measures_layers[0].weight.data.size()[0])
        self.measures_layers[0].weight.data.uniform_(-f3, f3)
        self.measures_layers[0].bias.data.uniform_(-f3, f3)

        f4 = 1. / np.sqrt(self.measures_layers[1].weight.data.size()[0])
        self.measures_layers[1].weight.data.uniform_(-f4, f4)
        self.measures_layers[1].bias.data.uniform_(-f4, f4)

        self.action_value = nn.Linear(self.n_actions, self.fc2 * 2)
        self.q = nn.Linear(self.fc2 * 2, 1)

        # weights for action_value and q layer
        f5 = 0.003
        self.q.weight.data.uniform_(-f5, f5)
        self.q.bias.data.uniform_(-f5, f5)

        f6 = 1. / np.sqrt(self.action_value.weight.data.size()[0])
        self.action_value.weight.data.uniform_(-f6, f6)
        self.action_value.bias.data.uniform_(-f6, f6)

        self.optim = optim.Adam(self.parameters(), lr=beta,weight_decay=0.01)
        self.device = T.device("cuda" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, img_state, measure_state,action):
        """
        compute for Image
        compute for measures
        concate and pass to fc layers
        """
        #TODO push the variables to gpu
        for layer in self.cnn_layers:
            img_state = layer(img_state)

        img_state = img_state.view(img_state.size(0),-1)

        for layer in self.fc_layers:
            img_state = layer(img_state)

        for layer in self.measures_layers:
            measure_state = layer(measure_state)
        #TODO remove Squzee from the test img_state
        img_measure_state = T.cat([img_state, measure_state], dim=-1)

        action_value = self.action_value(action)
        #print(action_value)
        state_action_value = F.relu(T.add(img_measure_state,action_value))
        state_action_value = self.q(state_action_value)
        #print(state_action_value)


        return state_action_value

    def save_model_checkpoint(self):
        logging.info("Saving the model")
        T.save(self.state_dict(), self.chkt_file)

    def load_model_checkpoint(self):
        logging.info("Loading the model")
        self.load_state_dict(T.load(self.chkt_file))

    def save_best_checkpoint(self):
        logging.info("Saving the best model")
        c_f = os.path.join(self.chkt_dir, self.chkt_name + "_cn_best")
        T.save(self.state_dict(), c_f)


if __name__ == "__main__":
    """ac = ActionSpace()
    obs = ObservationSpace()
    observation = obs.observation_space.sample()
    actions = ac.action_spaces.sample()


    print(actions.shape)
    ac= ActorNetwork(measures_state_dims=4,alpha = 0.001,fc1=400,fc2 = 300,actions=2,chkpt_name="Actor")
    ac.forward(img_state=T.tensor(observation['camera'],dtype=T.float).unsqueeze(0),measure_state=T.tensor(observation['state'],dtype=T.float))

"""
    pass
