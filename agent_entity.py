import random

from logger import log

class Agent:
    def __init__(self):
        self.action_space = [-1, 0, 1]
        self.history_obs = []
        self.history_acts = []
        self.history_rwds = []

    def receive_observation(self, obs, delta_t):
        if len(obs) > 0:
            for i in range(len(obs)):
                log.logger.debug('[Agent][receives s['+ str(obs[i].id)+']] = '+''.join(str(obs[i].value)))
                #self.history_rwds.append()
            self.history_obs.append(obs[0]) # store latest observation
            return self.action_space[random.randint(1,3)-1]
        else:
            log.logger.debug('[Agent][does not receive any observation at this time point]')
            return None
