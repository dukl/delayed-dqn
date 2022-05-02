import random

import keras.models
import numpy as np

import DNN.dnn
from logger import log, logR
#from DQN_Model.RL_brain import DeepQNetwork
from DQN_Model.DQN_Keras import DQN
from copy import deepcopy
from environment import ENV


class Agent:
    def __init__(self):
        self.action_space = [-1, 0, 1]
        self.history_obs = []
        self.history_acts = []
        self.history_rwds = []
        self.model = DQN(
            3, 25, learning_rate=0.0001, reward_decay=0.9, e_greedy=0.9, replace_target_iter=200, memory_size=100000, batch_size=500
        )
        self.step = 0
        self.pending_state = None
        self.pending_action = None
        self.epison_reward = []
        self.index = 0
        self.reward_sum = 0
        self.isPredGT = False
        self.isPredDNN = False
    def reset(self):
        self.step = 0
        self.pending_state = None
        self.pending_action = None
        self.index = 0
        self.reward_sum = 0
        self.isPredGT = False
        self.isPredDNN = False

    def receive_observation(self, obs, delta_t):
        if len(obs) > 0:
            for i in range(len(obs)):
                log.logger.debug('[Agent][receives s['+ str(obs[i].id)+']] = '+''.join(str(obs[i].value)))
            self.step += 1
            self.index += 1
            log.logger.debug('[Agent][Step: %d]' % (self.step))
            if obs[0].reward.value is not None:
                self.reward_sum += obs[0].reward.value
                logR.logger.debug('One Step Reward (%d): %f' % (self.index, obs[0].reward.value))
            #else:
            #    self.epison_reward.append(self.reward_sum)
            #    logR.logger.debug('Epision Reward %d: %f' % (len(self.epison_reward), self.epison_reward[-1]))
            #    self.reward_sum = 0
            #    self.index = 0
            #if (self.step >= 50) and (self.step % 5 ==0):
            #    log.logger.debug('[DQN][Training]')
            #    self.model.learn()

            delay = np.random.uniform(0,1,None)
            if self.isPredGT is True:
                log.logger.debug('[ENV][GT][changing over time]')
                log.logger.debug('[ENV][GT][Current State] %s' % str(obs[0].value))
                reward_bias = 0
                envGT = obs[0].env
                reward_bias += envGT.model.step(None, None, 0, delay, envGT.model.it_time + 2)
                obsV, reward = envGT.get_obs_rewards(obs[0].inputMsgs, None, reward_bias, 0)
                log.logger.debug('[ENV][GT][Predicted State] %s' % str(obsV))
                del envGT
                obs[0].value = obsV
            elif self.isPredDNN is True:
                model = DNN.dnn.DNNModel(25,25)
                model = keras.models.load_model('DNN/saved_model', compile=False)
                model.load_weights('DNN/my_model_weights.h5', by_name=True)
                model.compile(loss='mean_squared_error', optimizer=optimizers.Adam(0.0001))
                obsV = model.predict(np.array(obs[0].value).reshape(1,25)[0])
                obs[0].value = obsV
            observation = np.array(obs[0].value).reshape(1,25)[0]
            log.logger.debug('[reshape obs before: %s]' % str(observation))
            observation[observation==0] = 0.01
            log.logger.debug('[reshape obs after : %s]' % str(observation))

            if self.pending_action is not None:
                log.logger.debug('Transition: \n%s,[%d,%f],%s' % (str(self.pending_state), self.pending_action, obs[0].reward.value, str(observation)))
                self.model.store_transition(self.pending_state, self.pending_action, obs[0].reward.value, observation)
                if (self.model.memory_counter >= 600) and (self.model.memory_counter % 100 == 0):
                    logR.logger.debug('Training ...')
                    log.logger.debug('Training  ...')
                    self.model.learn()
            self.pending_state = observation
            action = self.model.choose_action(self.pending_state)
            self.pending_action = action
            logR.logger.debug('Agent generates action (%d, %d)' % (action, self.action_space[action]))
            return self.action_space[action], delay
        else:
            log.logger.debug('[Agent][does not receive any observation at this time point]')
            return None, None
