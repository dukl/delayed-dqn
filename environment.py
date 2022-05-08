import copy

import numpy as np
import tensorflow as tf
from logger import log
from flowModel import FM
from rewardModel import RM
from entities import AmfEntity
import math

AMF_CAPACITY = 300

class ENV(object):
    def __init__(self, iscopy = False, other=None):
        self.old_ob  = None
        self.model = FM(0, iscopy)
        self.deep_cp_attr = None
        self.action_seq = []
        self.oldInputMsg = 0
        if other is not None:
            self.__dict__ = copy.deepcopy(other.__dict__)

    def __deepcopy__(self, memodict={}):
        cpyobj = type(self)()
        cpyobj.deep_cp_attr = copy.deepcopy( memodict)
        return cpyobj


    def reset(self):
        self.action_seq.clear()
        del self.model
        self.model = FM(0, iscopy=False)
    def step(self, action):
        print('aaa')
    def get_obs_rewards(self, n_input_msgs, acts, reward_bais, id):
        log.logger.debug('[ENV][get_obs_rewards]')
        #self.model.amfList.sort(key=lambda AmfEntity: AmfEntity.id, reverse=False)
        obs = [n_input_msgs+len(self.model.msgInRISE), self.model.usefulUpRoad, len(self.model.msgDnOnRoad), self.model.Delay_Up_Link, self.model.Delay_Up_Link]
        n_total_msgs = self.oldInputMsg
        n_amf_inst = 0
        cpus, msgs = [], []
        for i in range(self.model.MAX_AMF_INST):
            if i < len(self.model.amfList):
                n_amf_inst += 1
                n_total_msgs += self.model.amfList[i].n_msgs
                cpus.append(self.model.amfList[i].n_cpu_cores)
                msgs.append(self.model.amfList[i].n_msgs)
            else:
                cpus.append(0)
                msgs.append(0)
        obs = obs + cpus + msgs
        #for i in range(self.model.MAX_AMF_INST):
        #    if i < len(self.model.amfList):
        #        if self.model.amfList[i].close == True:
        #            obs += [0, 0]
        #        else:
        #            n_amf_inst += 1
        #            n_total_msgs += self.model.amfList[i].n_msgs
        #            obs += [self.model.amfList[i].n_msgs, self.model.amfList[i].n_cpu_cores*5]
        #    else:
        #        obs += [0,0]
        if acts == None:
            self.oldInputMsg = n_input_msgs
            reward = RM(id, None, None, id)
            return obs, reward
        n_total_msgs += len(self.model.msgInRISE) + self.model.usefulUpRoad + len(self.model.msgDnOnRoad)
        running_time = acts[-1].current_status
        delta_x = 0
        if acts[-1].value == 1:
            if n_amf_inst == 10:
                capacity = n_total_msgs / (n_amf_inst)
                capacity -= 20 * (1 - running_time)
                if capacity < 0:
                    capacity = 0
                delta_x = (0.90 - capacity / AMF_CAPACITY) / 0.9 - (n_amf_inst + 1) / 20 - 1 # only support maximum 20 AMFs
            elif n_amf_inst < 10:
                capacity = n_total_msgs / (n_amf_inst + 1)
                capacity -= 20 * (1 - running_time)
                if capacity < 0:
                    capacity = 0
                delta_x = (0.90 - capacity/AMF_CAPACITY)/0.9 - (n_amf_inst + 1)/20 # only support maximum 20 AMFs
        if acts[-1].value == 0:
            capacity = n_total_msgs / (n_amf_inst)
            capacity -= 20 * (1 - running_time)
            if capacity < 0:
                capacity = 0
            delta_x = (0.90 - capacity / AMF_CAPACITY) / 0.9 - (n_amf_inst) / 20  # only support maximum 20 AMFs
        if acts[-1].value == -1:
            if n_amf_inst == 1:
                capacity = n_total_msgs / (n_amf_inst)
                capacity -= 20 * (1 - running_time)
                if capacity < 0:
                    capacity = 0
                delta_x = (0.90 - capacity / AMF_CAPACITY) / 0.9 - (n_amf_inst) / 20 - 1 # only support maximum 20 AMFs
            elif n_amf_inst > 1:
                capacity = n_total_msgs / (n_amf_inst - 1)
                capacity -= 20 * (1 - running_time)
                if capacity < 0:
                    capacity = 0
                delta_x = (0.90 - capacity / AMF_CAPACITY) / 0.9 - (n_amf_inst - 1) / 20  # only support maximum 20 AMFs
        rwdV = 0
        delta_x += 1 # 10 / 20
        log.logger.debug('[REWARD][%d, %d, %d, %f, %f, %f]' % (acts[-1].value, n_total_msgs, n_amf_inst, 1- running_time, capacity, delta_x))

        #if delta_x >= 0:
        #    rwdV = 1/(delta_x + 0.001)
        #else:
        #    rwdV = -math.exp(-delta_x)
        rwdV = delta_x + reward_bais
        rwdV = (rwdV - (-0.5)) / (1.95 - (-0.5))

        log.logger.debug('[REWARD][return reward: %f]' % (rwdV))
        reward = RM(id, rwdV, acts[-1].id, id)
        return obs, reward

    def send_observation_no_delay(self, acts, delta_t, n_input_msgs):
        reward_bias = 0
        if delta_t == 0:
            return self.get_obs_rewards(n_input_msgs, None, reward_bias, delta_t)
        if len(acts) > 0:
            self.action_seq.append(acts[0].value)
            obs_, reward = self.get_obs_rewards(n_input_msgs, acts, reward_bias, delta_t)
            log.logger.debug('[ENV][action %d] = %d is being executed' % (acts[0].id, acts[0].value))
            reward_bias += self.model.step(acts[0].value, acts[0].id, acts[0].id + acts[0].current_status, delta_t, delta_t)
            log.logger.debug('[ENV][action %d] = %d is executed' % (acts[0].id, acts[0].value))
            obs, reward_ = self.get_obs_rewards(n_input_msgs, acts, reward_bias, delta_t)
            reward.value += reward_bias
            return obs, reward
        else:
            return None,None

    def send_observation(self, acts, delta_t, n_input_msgs):
        reward_bias = 0
        if delta_t == 0:
            return self.get_obs_rewards(n_input_msgs, None, reward_bias, delta_t)
        if len(acts) > 0:
            self.action_seq.append(acts[0].value)
            reward_bias += self.model.step(None, None, acts[0].old_obs_id, acts[0].id + acts[0].current_status, delta_t)
            obs_, reward = self.get_obs_rewards(n_input_msgs, acts, reward_bias, delta_t)
            obs_[0] = len(self.model.msgInRISE)
            log.logger.debug('Real Obs: %s' % (str(obs_)))
            reward_bias += self.model.step(acts[0].value, acts[0].id, acts[0].id + acts[0].current_status, delta_t, delta_t)
            obs, reward_ = self.get_obs_rewards(n_input_msgs, acts, reward_bias, delta_t)
            n = len(self.model.inputMsgs)
            log.logger.debug('DUKL: %d' % (n))
            for i in range(n-1):
                del self.model.inputMsgs[0]
                del self.model.flag[0]
            return obs, reward
        else:
            return None,None




