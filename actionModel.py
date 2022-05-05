from logger import log
import numpy as np
import math

class AM:
    def __init__(self, action, id, delay, old_obs_id):
        self.value = action
        self.id = id
        self.old_obs_id = old_obs_id
        self.time_step = 0
        self.current_status = delay
        self.time_left_in_env = 0
    def check_status(self,delta_t):
        if self.id == delta_t:
            log.logger.debug('[Action][a[%d] = %d is newly generated [%f]' % (self.id, self.value, self.current_status))
            return 'NewAct'
        self.time_step += 1
        time_diff = self.current_status - self.time_step
        if time_diff <= 0:
            self.time_left_in_env = math.fabs(time_diff + 1)
            log.logger.debug('[Action][a[%d] arrived at the Env' % (self.id))
            return 'Arrived'
        else:
            log.logger.debug('[Action][a[%d] is on the road][%f]' % (self.id, time_diff))
            return 'OnRoad'