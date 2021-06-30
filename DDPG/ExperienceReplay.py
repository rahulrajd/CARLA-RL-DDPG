import numpy as np

class ExperienceReplay():
    def __init__(self, max_size, camera_shape, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0

        self.state_camera_memory = np.zeros((self.mem_size, *camera_shape))
        self.state_measurement_memory = np.zeros((self.mem_size,*input_shape))

        self.new_state_camera_memory = np.zeros((self.mem_size, *camera_shape))
        self.new_measurement_memory = np.zeros((self.mem_size,*input_shape))

        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)

        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_camera_memory[index] = state['camera']
        self.state_measurement_memory[index] = state['state']
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_camera_memory[index] = state_['camera']
        self.new_measurement_memory[index] = state_['state']
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        states = {'camera':None,
                 'state':None}
        states_ = {
            'camera':None,
            'state':None
        }

        batch = np.random.choice(max_mem, batch_size)
        states['camera'] = self.state_camera_memory[batch]
        states['state'] = self.state_measurement_memory[batch]

        #states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_['camera'] = self.new_state_camera_memory[batch]
        states_['state'] = self.new_measurement_memory[batch]

        #states_ = self.new_state_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones