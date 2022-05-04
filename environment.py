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
        n_total_msgs = 0
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
        log.logger.debug('[REWARD][return reward: %f]' % (delta_x + reward_bais))
        #if delta_x >= 0:
        #    rwdV = 1/(delta_x + 0.001)
        #else:
        #    rwdV = -math.exp(-delta_x)
        rwdV = delta_x + reward_bais
        rwdV = (rwdV - (-0.5)) / (1.95 - (-0.5))


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
            reward_bias += self.model.step(acts[0].value, acts[0].id, 0, 1, delta_t)
            log.logger.debug('[ENV][action %d] = %d is executed' % (acts[0].id, acts[0].value))
            obs, reward_ = self.get_obs_rewards(n_input_msgs, acts, reward_bias, delta_t)
            reward.value += reward_bias
            return obs, reward
        else:
            return None,None
    def send_observation(self, acts, delta_t, n_input_msgs):
        log.logger.debug('[ENV][send_observation]')
        reward_bais = 0
        if delta_t == 0:
            return self.get_obs_rewards(n_input_msgs, None, reward_bais, delta_t)
        tp = [0]
        if len(acts) > 0:
            self.action_seq.append(acts[0].value)
            #if len(acts) == 1:
            #    reward_bais += self.model.step(None, None, 0, 1, delta_t)
            #    obs, reward = self.get_obs_rewards(n_input_msgs, acts, reward_bais)
            #    reward_bais += self.model.step(acts[0].value, None, 0, 1, delta_t)
            acts.sort(key=lambda AM: AM.time_left_in_env, reverse=False)
            for i in range(len(acts)):
                log.logger.debug('[ENV][receive a[%d]=%d (time in advance : %f)]' % (acts[i].id, acts[i].value, acts[i].time_left_in_env))
                tp.append(acts[i].time_left_in_env)
            #tp.append(1)
            index = 0
            for i in range(len(acts)):
                if i == 0:
                    reward_bais += self.model.step(None, None, tp[index], tp[index + 1], delta_t)
                    continue
                else:
                    log.logger.debug('[ENV][action %d] = %d is being executed' % (acts[index].id, acts[index].value))
                    reward_bais += self.model.step(acts[index].value, acts[index].id, tp[index], tp[index+1], delta_t)
                    log.logger.debug('[ENV][action %d] = %d is executed with bias %d' % (acts[index].id, acts[index].value, reward_bais))
                index += 1
            obs_, reward = self.get_obs_rewards(0, acts, reward_bais, delta_t)
            log.logger.debug('[ENV][Get Current States]\n %s' % (str(obs_)))
            #self.model.check_delete_AMF_inst()
            log.logger.debug('[ENV][action %d] = %d is being executed' % (acts[-1].id, acts[-1].value))
            reward_bais += self.model.step(acts[-1].value, acts[-1].id, acts[-1].time_left_in_env, 1, delta_t)
            log.logger.debug('[ENV][action %d] = %d is executed with bias %d' % (acts[-1].id, acts[-1].value, reward_bais))
            obs, reward_ = self.get_obs_rewards(n_input_msgs, acts, reward_bais, delta_t)
            reward.value += reward_bais

            return obs, reward
        else:
            log.logger.debug('[ENV][does not receive any actions at this time point]')
            #reward_bais += self.model.step(None, None, 0, 1, delta_t)
            #obs, reward = self.get_obs_rewards(n_input_msgs, None, reward_bais, delta_t)
            #self.model.check_delete_AMF_inst()
            return None, None


