import numpy as np
import tensorflow as tf
from logger import log
from flowModel import FM
from rewardModel import RM
from entities import AmfEntity
import math

class ENV():
    def __init__(self):
        self.old_ob  = None
        self.model = FM(0)
    def reset(self):
        print('aaa')
    def step(self, action):
        print('aaa')
    def get_obs_rewards(self, n_input_msgs, acts, reward_bais, id):
        self.model.amfList.sort(key=lambda AmfEntity: AmfEntity.id, reverse=False)
        obs = [n_input_msgs]
        n_total_msgs = 0
        n_amf_inst = 0
        for i in range(self.model.MAX_AMF_INST):
            if i < len(self.model.amfList):
                if self.model.amfList[i].close == True:
                    obs += [0, 0]
                else:
                    n_amf_inst += 1
                    n_total_msgs += self.model.amfList[i].n_msgs
                    obs += [self.model.amfList[i].n_msgs, self.model.amfList[i].n_cpu_cores*5]
            else:
                obs += [0,0]
        if acts == None:
            reward = RM(id, None, None, id)
            return obs, reward
        delta_x = 0
        if acts[-1].value == 1:
            delta_x = 0.90 - ((n_input_msgs - n_total_msgs/n_amf_inst)/(n_amf_inst+1) + n_total_msgs/n_amf_inst)/200
        if acts[-1].value == 0:
            delta_x = 0.90 - ((n_input_msgs + n_total_msgs)/n_amf_inst)/200
        if acts[-1].value == -1:
            delta_x = 0.90 - ((n_input_msgs + n_total_msgs*(n_amf_inst-1)/n_amf_inst)/(n_amf_inst-1))/200
        rwdV = 0
        if delta_x >= 0:
            rwdV = 1/(delta_x + 0.001)
        else:
            rwdV = -math.exp(-delta_x)
        rwdV += reward_bais
        reward = RM(id, rwdV, acts[-1].id, id)
        return obs, reward
    def send_observation(self, acts, delta_t, n_input_msgs):
        reward_bais = 0
        if delta_t == 0:
            return self.get_obs_rewards(n_input_msgs, None, reward_bais, delta_t)
        tp = [0]
        if len(acts) > 0:
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
                    reward_bais += self.model.step(acts[index].value, acts[index].id, tp[index], tp[index+1], delta_t)
                index += 1
            obs_, reward = self.get_obs_rewards(n_input_msgs, acts, reward_bais, delta_t)
            reward_bais += self.model.step(acts[-1].value, acts[-1].id, acts[-1].time_left_in_env, 1, delta_t)
            obs, reward_ = self.get_obs_rewards(n_input_msgs, acts, reward_bais, delta_t)
            reward.value += reward_bais
            return obs, reward
        else:
            log.logger.debug('[ENV][does not receive any actions at this time point]')
            reward_bais += self.model.step(None, None, 0, 1, delta_t)
            return self.get_obs_rewards(n_input_msgs, None, reward_bais, delta_t)


