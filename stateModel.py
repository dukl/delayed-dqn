import numpy as np
from logger import log




class SM:
    def __init__(self, state, id):
        self.value = state
        self.trans_delay = np.random.uniform(1,4,None)
        self.current_status = self.trans_delay
        self.id = id
        self.time_step = 0
        log.logger.debug('[State][s[%d] is newly generated [%f]' % (self.id, self.current_status))
    def check_status(self):
        self.time_step += 1
        time_diff = self.current_status - self.time_step
        if time_diff <= 0:
            log.logger.debug('[State][s[%d] arrived at the Agent]', self.id)
            return 'Arrived'
        else:
            log.logger.debug('[State][s[%d] is on the road][%f]' % (self.id, time_diff))
            return 'OnRoad'