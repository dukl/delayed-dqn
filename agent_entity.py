import random

import numpy as np

from logger import log
from DQN_Model.RL_brain import DeepQNetwork

class Agent:
    def __init__(self):
        self.action_space = [-1, 0, 1]
        self.history_obs = []
        self.history_acts = []
        self.history_rwds = []
        self.model = DeepQNetwork(
            3, 25, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9, replace_target_iter=200, memory_size=2000
        )
        self.step = 0
        self.pending_state = None
        self.pending_action = None

    def receive_observation(self, obs, delta_t):
        if len(obs) > 0:
            for i in range(len(obs)):
                log.logger.debug('[Agent][receives s['+ str(obs[i].id)+']] = '+''.join(str(obs[i].value)))
            self.step += 1
            if (self.step > 200) and (self.step % 5 ==0):
                self.model.learn()
            if self.pending_action is not None:
                self.model.store_transition(self.pending_state, self.pending_action, obs[0].reward.value, np.array(obs[0].value).reshape(1,25))
            self.pending_state = np.array(obs[0].value).reshape(1, 25)
            action = self.model.choose_action(self.pending_state)
            self.pending_action = action

            return self.action_space[action]
        else:
            log.logger.debug('[Agent][does not receive any observation at this time point]')
            return None
