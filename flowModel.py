import random
from queue import Queue
from logger import log
import entities
from entities import Msgs
from entities import AmfEntity
import numpy as np

class FM:
    def __init__(self, n_ue_reqs):
        self.total_ue_reqs = n_ue_reqs
        self.n_ue_reqs = n_ue_reqs
        self.n_amf_insts = 1
        self.it_time = 0
        self.time_interval = 0.01
        self.Delay_Up_Link = 0.5
        self.Delay_Down_Link = 0.5
        self.msgUpOnRoad = Queue(maxsize=0)
        self.msgDnOnRoad = Queue(maxsize=0)
        self.msgInRISE   = Queue(maxsize=0)
        self.AMFIndex = 0
        self.amfList  = []
        self.numAMF   = 1
        self.cpuInAMF = []
        self.initialize()
        self.add_new_action = False
        self.MAX_AMF_INST = 10
        self.usefulUpRoad = 0

    def update_ue_reqs_every_time_step(self, n_ue_reqs):
        for i in range(n_ue_reqs):
            message = Msgs(self.total_ue_reqs + i + 1, 1, 'RegistrationRequest')
            self.msgInRISE.put(message)
        self.total_ue_reqs += n_ue_reqs

    def initialize(self):
        for i in range(self.numAMF):
            self.amfList.append(AmfEntity(np.random.uniform(1,6,None), i, 0))
        for i in range(self.n_ue_reqs):
            message = Msgs(i+1, 1, 'RegistrationRequest')
            self.msgInRISE.put(message)

    def check_available_AMF_Inst(self):
        self.AMFIndex = 0
        min_n_msgs = 10000
        for i in range(len(self.amfList)):
            if self.amfList[i].close == True:
                continue
            elif self.amfList[i].n_msgs < min_n_msgs:
                min_n_msgs = self.amfList[i].n_msgs
                self.AMFIndex = i

    def check_close_which_AMF_Inst(self):
        self.AMFIndex = 0
        max_n_msgs = 10000
        for i in range(len(self.amfList)):
            if self.amfList[i].n_msgs > max_n_msgs:
                max_n_msgs = self.amfList[i].n_msgs
                self.AMFIndex = i
        if len(self.amfList) > 1:
            log.logger.debug('[FlowModel][AMF %d] has beed closed (n_msgs %d)' % (self.amfList[self.AMFIndex].id, self.amfList[self.AMFIndex].n_msgs))
            msgs = self.amfList[self.AMFIndex].message_queue
            del self.amfList[self.AMFIndex]
            msgs_size = int(msgs.qsize() / len(self.amfList))
            for i in range(len(self.amfList)):
                for j in range(msgs_size):
                    self.amfList[i].message_queue.put(msgs.get())
                log.logger.debug('[FlowModel][AMF %d][%d msgs]' % (self.amfList[i].id, self.amfList[i].message_queue.qsize()))
                #self.amfList[self.AMFIndex].close = True

    def check_delete_AMF_inst(self):
        log.logger.debug('[FlowModel][check_delete_AMF_inst ...]')
        index = []
        for i in range(len(self.amfList)):
            if self.amfList[i].close == True:
                index.append(i)
        log.logger.debug('[FlowModel][Killing amf inst with index: ' + ''.join(str(index)) + ']')
        for i in range(len(index)):
            log.logger.debug('[FlowModel][AMF %d] has beed killed' % self.amfList[index[len(index)-i-1]].id)
            del self.amfList[index[len(index) - i - 1]]

    def action_execution(self, action, delta_t):
        reward_bias = 0
        # log.logger.debug('[FlowModel][Action a[%d] = %d is executed]' % (id, action))
        if action == 1:
            self.numAMF += 1
            #if self.numAMF > self.MAX_AMF_INST:
            #    log.logger.debug('Maximum Number of AMF Instance is %d, ignore this action' % (self.MAX_AMF_INST))
            #    reward_bias -= 10
            #else:
            self.amfList.append(AmfEntity(np.random.uniform(2, 4, None), self.numAMF - 1, delta_t))
        if action == -1:
            if len(self.amfList) == 1:
                reward_bias -= 100
                log.logger.debug('Number of AMF instance is less than 1, so missed this action, return reward -100')
            else:
                self.check_close_which_AMF_Inst()
        return reward_bias

    def step(self, action, id, tpS, tpE, delta_t):
        log.logger.debug('[ENV][step] [%f ~ %f]' % (self.it_time, tpE + delta_t - 1))
        reward = 0
        if action == None:
            log.logger.debug('[FlowModel][No action is executed]')

        self.add_new_action = False

        while self.it_time < (tpE + delta_t - 1):
            if action != None and self.it_time >= tpS + delta_t - 1 and self.add_new_action == False:
                self.add_new_action = True
                reward = self.action_execution(action, delta_t)
            self.it_time += self.time_interval
            log.logger.debug('[FlowModel][At time point: %f]' % (self.it_time))
            #log.logger.debug('[FlowModel][No. of Msgs in RISE: %d]' % (self.msgInRISE.qsize()))
            if self.msgInRISE.qsize() > 0:
                message = self.msgInRISE.get()
                #log.logger.debug('Message (%d, %d, %s) has been out of RISE, waiting to be AMF' % (message.ue_id, message.msg_id, message.msgType))
                self.msgUpOnRoad.put(message)
                self.usefulUpRoad += 1
            else:
                message = Msgs(0,0,'NULL')
                #log.logger.debug('No Messages in RISE already')
                self.msgUpOnRoad.put(message)
            if self.msgUpOnRoad.qsize() >= self.Delay_Up_Link/self.time_interval:
                #log.logger.debug('msgUpOnRoad is full ...')
                message = self.msgUpOnRoad.get()
                if message.msg_id > 0:
                    self.usefulUpRoad -= 1
                    self.check_available_AMF_Inst()
                    #log.logger.debug('Message (%d, %d, %s) has arieved at AMF (%d), to be processed' % (message.ue_id, message.msg_id, message.msgType, self.AMFIndex))
                    self.amfList[self.AMFIndex].stateTrans(message.ue_id, message.msg_id, message.msgType, self.time_interval)
                    for i in range(len(self.amfList)):
                        if i == self.AMFIndex:
                            continue
                        else:
                            self.amfList[i].stateTrans(0,0,'NULL',self.time_interval)
                    #self.AMFIndex = (self.AMFIndex + 1) % self.numAMF
                else:
                    #log.logger.debug('No message is put into AMFs')
                    for i in range(len(self.amfList)):
                        self.amfList[i].stateTrans(0,0,'NULL', self.time_interval)
            else:
                for i in range(len(self.amfList)):
                    self.amfList[i].stateTrans(0, 0, 'NULL', self.time_interval)
            for i in range(len(self.amfList)):
                if self.amfList[i].newMsg.msg_id > 0:
                    #log.logger.debug('AMF (%d) generate new message (%d, %d, %s) into msgDnOnRoad (%d)' % (i, self.amfList[i].newMsg.ue_id, self.amfList[i].newMsg.msg_id, self.amfList[i].newMsg.msgType, self.msgDnOnRoad.qsize()))
                    message = Msgs(0,0,'NULL')
                    message.ue_id = self.amfList[i].newMsg.ue_id
                    message.msg_id = self.amfList[i].newMsg.msg_id
                    message.msgType = self.amfList[i].newMsg.msgType
                    self.msgDnOnRoad.put(message)
            if self.msgDnOnRoad.qsize() >= self.Delay_Down_Link / self.time_interval:
                #log.logger.debug('msgDnOnRoad is full ...')
                message = self.msgDnOnRoad.get()
                #log.logger.debug(message.ue_id, message.msg_id, message.msgType)
                #log.logger.debug('msgDnOnRoad.get: message (%d, %d, %s)' % (message.ue_id, message.msg_id, message.msgType))
                if message.msg_id > 0 and message.msg_id < 6:
                    back_message = Msgs(message.ue_id, message.msg_id + 1, entities.reqMsgs[message.msg_id])
                    #log.logger.debug('Message (%d, %d, %s) has been sent back to RISE, RISE generate new message (%d, %d, %s)' % (message.ue_id, message.msg_id, message.msgType, back_message.ue_id, back_message.msg_id, back_message.msgType))
                    self.msgInRISE.put(back_message)
                #else:
                    #log.logger.debug('RISE dosennot genearte new message any more')
            #self.check_delete_AMF_inst()
            log.logger.debug('Statics (%d) AMFs ...' % (len(self.amfList)))
            for i in range(len(self.amfList)):
                if self.amfList[i].close == True:
                    log.logger.debug('AMF (%d) has been closed' % (self.amfList[i].id))
                log.logger.debug('AMF (%d) has (%d) messages' % (self.amfList[i].id, self.amfList[i].n_msgs))
        return reward





