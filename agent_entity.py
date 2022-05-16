import random

import keras.models
import numpy as np

import DNN.dnn
from logger import log, logR
#from DQN_Model.RL_brain import DeepQNetwork
from DQN_Model.DQN_Keras import DQN
from copy import deepcopy
from environment import ENV
from DQN_Model.dqn_agents import DDQNPlanningAgent, DDQNAgent

class Agent:
    def __init__(self):
        self.action_space = [-1, 0, 1]
        self.history_obs = []
        self.history_acts = []
        self.history_rwds = []
        #self.model = DQN(
        #    3, 25, learning_rate=0.001, reward_decay=0.9, e_greedy=0.9, replace_target_iter=20, memory_size=100000, batch_size=1280
        #)
        self.model = DDQNPlanningAgent(25, 3, False, False, 1, 0.001, 0.999, 0.001, 1, False, True, None, True)
        #self.model = DDQNAgent(27, 3, False, False, 0, 0.001, 0.999, 0.001, 1, False, True)
        #self.model = DDQNAgent(25, 3, False, False, 0, 0.001, 0.999, 0.001, 1, False, True)
        self.step = 0
        self.pending_state = None
        self.pending_state_next = None
        self.pending_action = None
        self.epison_reward = []
        self.index = 0
        self.reward_sum = 0
        self.isPredGT = False
        self.isPredDNN = False
        self.act_buf = []
        self.obs_buf = []
        self.step_num = 0

    def reset(self):
        self.step_num = 0
        self.step = 0
        self.pending_state = None
        self.pending_action = None
        self.index = 0
        self.reward_sum = 0
        self.isPredGT = False
        self.isPredDNN = False
        self.act_buf.clear()
        self.obs_buf.clear()
        self.model.clear_action_buffer()

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

            delay = np.random.uniform(1,2,None)
            if self.isPredGT is True:
                log.logger.debug('[ENV][GT][changing over time]')
                log.logger.debug('[ENV][GT][Current State] %s' % str(obs[0].value))
                log.logger.debug(
                    '------------------------------------------------- GT Start--------------------------------------')
                reward_bias = 0
                envGT = obs[0].env
                reward_bias += envGT.model.step(None, None, obs[0].id, delta_t + delay, envGT.model.it_time + 2)
                obsV, reward = envGT.get_obs_rewards(obs[0].inputMsgs, None, reward_bias, 0)
                log.logger.debug('[ENV][GT][Predicted State] %s' % str(obsV))
                del envGT
                obs[0].value = obsV
                log.logger.debug(
                    '------------------------------------------------- GT End--------------------------------------')
            elif self.isPredDNN is True:
                log.logger.debug(
                    '------------------------------------------------- DNN Start--------------------------------------')
                model = DNN.dnn.DNNModel(25,25)
                model = keras.models.load_model('DNN/saved_model', compile=False)
                model.load_weights('DNN/my_model_weights.h5', by_name=True)
                #model.compile(loss='mean_squared_error', optimizer=optimizers.Adam(0.0001))
                obsV = model.predict(np.array(obs[0].value).reshape(25,1))
                log.logger.debug('[ENV][GT][Predicted State] %s' % str(obsV))
                obs[0].value = obsV[0]
                log.logger.debug(
                    '------------------------------------------------- DNN End--------------------------------------')
            observation = np.array(obs[0].value).reshape(1,25)[0]
            log.logger.debug('[reshape obs before: %s]' % str(observation))
            #observation[observation==0] = 0.01
            _range = np.max(observation) - np.min(observation)
            norm_obs = (observation - np.min(observation))/_range
            norm_obs[norm_obs==0] = 0.0001
            log.logger.debug('[reshape obs after : %s]' % str(norm_obs))

            ########### ARGUMENT ############
            #if self.pending_state is not None:
            #    next_obs = np.concatenate((norm_obs, self.act_buf))
            #    self.model.memorize(self.pending_state, self.pending_action, obs[0].reward.value, next_obs, False)
            #obs_for_act = np.concatenate((norm_obs, self.act_buf))
            #self.pending_state = obs_for_act
            #obs_for_act = np.reshape(obs_for_act, [1,27])
            #action = self.model.act(obs_for_act, eval=False)
            #self.pending_action = action
            #del self.act_buf[0]
            #self.act_buf.append(action)
            ############# FM ######################
            if self.pending_state is not None:
                self.model.memorize(self.pending_state, self.pending_action, obs[0].reward.value, norm_obs, False)
            action = self.model.act(norm_obs, self.act_buf, eval=False)
            self.pending_action = self.act_buf[0]
            del self.act_buf[0]
            self.act_buf.append(action)
            self.pending_state = norm_obs


            self.step_num += 1
            if self.step_num % 200 == 0:
                self.model.update_target_model()
            batch_size = 32
            if len(self.model.memory) > batch_size and self.step_num % 2 == 0:
                batch_loss_dict = self.model.replay(batch_size)

            return self.action_space[action], delay, obs[0].id
        else:
            log.logger.debug('[Agent][does not receive any observation at this time point]')
            #self.pending_action = 0
            #self.pending_state = [0 for _ in range(25)]
            self.act_buf.append(0)
            tmp = [0 for _ in range(25)]
            self.obs_buf.append(np.array(tmp))
            return self.action_space[1], 1.5, 0
