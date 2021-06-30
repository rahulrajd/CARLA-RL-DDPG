from models.gym_carla.gym_carla.observation_action_space import *
import numpy as np
import torch as T
import torch.nn.functional as F
from Network import ActorNetwork,CriticNetwork
#from env.observation_action_space import *
from OUA import OUActionNoise
from gym import spaces
from ExperienceReplay import ExperienceReplay
#from torch.utils.tensorboard import SummaryWriter




class DDPGAgent():
    def __init__(self,alpha,beta,measure_dim,camera_input_dim,tau,n_actions,gamma=0.99,max_buffer_size = 5,f1 = 400,f2 = 300, batch_size=64):
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.tau = tau
        self.memory = ExperienceReplay(max_size=max_buffer_size,camera_shape=camera_input_dim,input_shape=measure_dim,n_actions=n_actions)
        self.oua_noise = OUActionNoise(mu=np.zeros(n_actions))
        self.batch_size = batch_size

        self.actor = ActorNetwork(alpha=self.alpha,measures_state_dims=measure_dim,fc1=f1,fc2=f2,actions=n_actions,chkpt_name="Actor")
        self.target_actor = ActorNetwork(alpha=self.alpha,measures_state_dims=measure_dim,fc1=f1,fc2=f2,actions=n_actions,chkpt_name="Target-Actor")
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.critic = CriticNetwork(beta=self.beta,measures_state_dims=measure_dim,fc1=f1,fc2=f2,actions=n_actions,chkpt_name="Critic")
        self.target_critic = CriticNetwork(beta=self.beta,measures_state_dims=measure_dim,fc1=f1,fc2=f2,actions=n_actions,chkpt_name="Target-Critic")
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.update_parameters(tau=1)

    def learn(self):
        if self.memory.mem_cntr < self.memory.mem_size:
            return
        states, actions, rewards, new_state, done = self.memory.sample_buffer(batch_size=64)
        img_state = T.tensor(states['camera'],dtype=T.float).to(self.actor.device)
        measure_state = T.tensor(states['state'],dtype=T.float).to(self.actor.device)
        actions = T.tensor(actions,dtype=T.float).to(self.actor.device)
        rewards = T.tensor(rewards,dtype=T.float).to(self.actor.device)
        new_camera_state = T.tensor(new_state['camera'],dtype=T.float).to(self.actor.device)
        new_measure_state = T.tensor(new_state['state'],dtype=T.float).to(self.actor.device)
        done = T.tensor(done).to(self.actor.device)

        target_actions = self.target_actor.forward(new_camera_state,new_measure_state)
        critic_value_ = self.target_critic.forward(new_camera_state,new_measure_state,target_actions)
        critic_value = self.critic.forward(img_state,measure_state,actions)
        critic_value_[done] = 0.0
        critic_value_ = critic_value_.view(-1)

        target = rewards + self.gamma*critic_value_
        target = target.view(self.batch_size,-1)

        self.critic.optim.zero_grad()
        critic_loss = F.mse_loss(target,critic_value)
        #writer.add_scalar("Critic Loss", critic_loss)
        critic_loss.backward()
        self.critic.optim.step()

        self.actor.optim.zero_grad()
        actor_loss = -self.critic.forward(img_state,measure_state,self.actor.forward(img_state,measure_state))
        actor_loss = T.mean(actor_loss)
        #writer.add_scalar("Actor Loss", actor_loss)
        actor_loss.backward()
        self.actor.optim.step()

        #writer.add_scalars("Actor and Critic Losses",{"Actor_loss":actor_loss,"Critic_loss":critic_loss})
        self.update_parameters(tau=self.tau)

    def select_action(self,observation):
        self.actor.eval()
        camera,state = observation['camera'],observation['state']
        camera = T.tensor([camera],dtype=T.float).to(self.actor.device)
        state = T.tensor([state], dtype=T.float).to(self.actor.device)
        mu = self.actor.forward(camera,state).to(self.actor.device)
        mu_p = mu + T.tensor(self.oua_noise(),dtype=T.float).to(self.actor.device)
        self.actor.train()
        return mu_p.cpu().detach().numpy()[0]

    def save_buffer_stream(self,state,action,reward,state_,done):
        self.memory.store_transition(state,action,reward,state_,done)

    def update_parameters(self,tau):
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        actor_state_dict = dict(actor_params)
        critic_state_dict = dict(critic_params)
        target_actor_dict = dict(target_actor_params)
        target_critic_dict = dict(target_critic_params)

        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + (1-tau)*target_critic_dict[name].clone()

        for name in actor_state_dict:
            actor_state_dict[name] = tau*actor_state_dict[name].clone() + (1-tau)*target_actor_dict[name].clone()

        self.target_critic.load_state_dict(critic_state_dict,strict=False)
        self.target_actor.load_state_dict(actor_state_dict,strict=False)

    def save_models(self):
        self.actor.save_model_checkpoint()
        self.critic.save_model_checkpoint()
        self.target_actor.save_model_checkpoint()
        self.target_critic.save_model_checkpoint()

    def load_models(self):
        self.actor.load_model_checkpoint()
        self.critic.load_model_checkpoint()
        self.target_actor.load_model_checkpoint()
        self.target_critic.load_model_checkpoint()


"""if __name__ == "__main__":
    q = ObservationSpace()
    s = DDPGAgent(alpha=0.001,beta=0.0001,measure_dim=4,)
    a = s.select_action(q.observation_space.sample())"""
