import numpy as np
import tensorflow as tf
from logger import log
from flowModel import FM

class ENV():
    def __init__(self):
        self.old_ob  = None
        self.model = FM(0)
    def reset(self):
        print('aaa')
    def step(self, action):
        print('aaa')
    def send_observation(self, acts, delta_t):
        if delta_t == 0:
            return [1,2,3,4], 2
        tp = []
        if len(acts) > 0:
            acts.sort(key=lambda AM: AM.time_left_in_env, reverse=False)
            for i in range(len(acts)):
                log.logger.debug('[ENV][receive a[%d]=%d (time in advance : %f)]' % (acts[i].id, acts[i].value, acts[i].time_left_in_env))
                tp.append(acts[i].time_left_in_env)
            tp.append(1)
            index = 0
            for i in range(len(acts)):
                self.model.step(acts[i].value, acts[i].id, tp[index], tp[index+1], delta_t)
                index += 1
        else:
            log.logger.debug('[ENV][does not receive any actions at this time point]')
            self.model.step(None, None, 0, 1, delta_t)
        return [1,2,3,4], 2

