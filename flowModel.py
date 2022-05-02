import random
from queue import Queue
from logger import log
import entities
from entities import Msgs
from entities import AmfEntity
import numpy as np
import copy
import math
import sys
sys.setrecursionlimit(10000)


class FM(object):
    def __init__(self, n_ue_reqs, iscopy=False):
        self.total_ue_reqs = n_ue_reqs
        self.n_ue_reqs = n_ue_reqs
        self.n_amf_insts = 1
        self.it_time = 0
        self.time_interval = 0.01
        self.Delay_Up_Link = 0.5
        self.Delay_Down_Link = 0.5
        self.msgUpOnRoad = []
        self.msgDnOnRoad = []
        self.msgInRISE   = []
        self.AMFIndex = 0
        self.amfList  = []
        self.numAMF   = 1
        self.cpuInAMF = []
        if iscopy is False:
            self.initialize()
        self.add_new_action = False
        self.MAX_AMF_INST = 10
        self.usefulUpRoad = 0
        self.amf_id = 1
        self.inputMsgs = []
        self.n_discard_msgs = 0
        self.n_request_msgs = 0


    def __deepcopy__(self, memodict={}):
        cpyobj = type(self)(0)
        cpyobj.deep_cp_attr = copy.deepcopy(memodict)
        return cpyobj

    def update_ue_reqs_every_time_step(self, n_ue_reqs):
        self.inputMsgs.append(n_ue_reqs)
        #for i in range(n_ue_reqs):
        #    message = Msgs(self.total_ue_reqs + i + 1, 1, 'RegistrationRequest')
        #    self.msgInRISE.append(message)
        #self.total_ue_reqs += n_ue_reqs

    def initialize(self):
        for i in range(self.numAMF):
            self.amfList.append(AmfEntity(np.random.uniform(1,6,None), i, 0))
        for i in range(self.n_ue_reqs):
            message = Msgs(i+1, 1, 'RegistrationRequest')
            self.msgInRISE.append(message)

    def check_available_AMF_Inst(self):
        self.AMFIndex = -1
        min_n_msgs = 10000
        for i in range(len(self.amfList)):
            if self.amfList[i].close == True or (len(self.amfList[i].message_queue)+1)>300*0.9:
                continue
            elif self.amfList[i].n_msgs < min_n_msgs:
                min_n_msgs = self.amfList[i].n_msgs
                self.AMFIndex = i
        if self.AMFIndex == -1:
            self.n_discard_msgs += 1

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
            self.numAMF -= 1
            msgs_size = int(len(msgs) / len(self.amfList))
            for i in range(len(self.amfList)):
                for j in range(msgs_size):
                    if len(self.amfList[i].message_queue) + 1 <= 300*0.9:
                        self.amfList[i].message_queue.append(msgs[0])
                        del msgs[0]
                log.logger.debug('[FlowModel][AMF %d][%d msgs]' % (self.amfList[i].id, len(self.amfList[i].message_queue)))
                #self.amfList[self.AMFIndex].close = True
            self.n_discard_msgs += len(msgs)

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
            self.amf_id += 1
            if self.numAMF >= self.MAX_AMF_INST:
                log.logger.debug('Maximum Number of AMF Instance is %d, ignore this action' % (self.MAX_AMF_INST))
                #reward_bias -= 1
            else:
                self.numAMF += 1
                self.amfList.append(AmfEntity(np.random.uniform(2, 4, None), self.amf_id - 1, delta_t))
        if action == -1:
            if len(self.amfList) == 1:
                #reward_bias -= 1
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
        self.add_input_msgs = [False, False]
        self.isDeleted = False

        while self.it_time < (tpE + delta_t - 1):
            if self.it_time < delta_t - 1 and self.add_input_msgs[0] is False and self.isDeleted is False and action is None:
                self.add_input_msgs[0] = True
                log.logger.debug('[FlowModel][Adding new messages --- 0]')
                for i in range(self.inputMsgs[0]):
                    message = Msgs(self.total_ue_reqs + i + 1, 1, 'RegistrationRequest')
                    self.msgInRISE.append(message)
                self.total_ue_reqs += self.inputMsgs[0]
            if self.it_time >= delta_t - 1 and self.it_time < tpE + delta_t -1 and self.add_input_msgs[1] is False and self.isDeleted is False and action is None:
                self.add_input_msgs[1] = True
                log.logger.debug('[FlowModel][Adding new messages --- 1]')
                for i in range(self.inputMsgs[1]):
                    message = Msgs(self.total_ue_reqs + i + 1, 1, 'RegistrationRequest')
                    self.msgInRISE.append(message)
                self.total_ue_reqs += self.inputMsgs[1]
            if self.add_input_msgs[0] and self.add_input_msgs[1]:
                log.logger.debug('[FlowModel][deleting old messages]')
                del self.inputMsgs[1]
                del self.inputMsgs[0]
                self.add_input_msgs = [False, False]
                self.isDeleted = True
            if action != None and self.it_time >= tpS + delta_t - 1 and self.add_new_action == False:
                self.add_new_action = True
                reward = self.action_execution(action, delta_t)
            self.it_time += self.time_interval
            log.logger.debug('[FlowModel][At time point: %f]' % (self.it_time))
            log.logger.debug('[FlowModel][No. of Msgs in RISE: %d]' % (len(self.msgInRISE)))
            if len(self.msgInRISE) > 0:
                message = self.msgInRISE[0]
                del self.msgInRISE[0]
                #log.logger.debug('Message (%d, %d, %s) has been out of RISE, waiting to be AMF' % (message.ue_id, message.msg_id, message.msgType))
                self.msgUpOnRoad.append(message)
                self.usefulUpRoad += 1
            else:
                message = Msgs(0,0,'NULL')
                #log.logger.debug('No Messages in RISE already')
                self.msgUpOnRoad.append(message)
            if len(self.msgUpOnRoad) >= self.Delay_Up_Link/self.time_interval:
                #log.logger.debug('msgUpOnRoad is full ...')
                message = self.msgUpOnRoad[0]
                del self.msgUpOnRoad[0]
                if message.msg_id > 0:
                    self.n_request_msgs += 1
                    self.usefulUpRoad -= 1
                    self.check_available_AMF_Inst()
                    #log.logger.debug('Message (%d, %d, %s) has arieved at AMF (%d), to be processed' % (message.ue_id, message.msg_id, message.msgType, self.AMFIndex))
                    for i in range(len(self.amfList)):
                        if i == self.AMFIndex:
                            self.amfList[self.AMFIndex].stateTrans(message.ue_id, message.msg_id, message.msgType, self.time_interval)
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
                    self.msgDnOnRoad.append(message)
            if len(self.msgDnOnRoad) >= self.Delay_Down_Link / self.time_interval:
                #log.logger.debug('msgDnOnRoad is full ...')
                message = self.msgDnOnRoad[0]
                del self.msgDnOnRoad[0]
                #log.logger.debug(message.ue_id, message.msg_id, message.msgType)
                #log.logger.debug('msgDnOnRoad.get: message (%d, %d, %s)' % (message.ue_id, message.msg_id, message.msgType))
                if message.msg_id > 0 and message.msg_id < 6:
                    back_message = Msgs(message.ue_id, message.msg_id + 1, entities.reqMsgs[message.msg_id])
                    #log.logger.debug('Message (%d, %d, %s) has been sent back to RISE, RISE generate new message (%d, %d, %s)' % (message.ue_id, message.msg_id, message.msgType, back_message.ue_id, back_message.msg_id, back_message.msgType))
                    self.msgInRISE.append(back_message)
                #else:
                    #log.logger.debug('RISE dosennot genearte new message any more')
            #self.check_delete_AMF_inst()
            log.logger.debug('Statics (%d) AMFs ...' % (len(self.amfList)))
            for i in range(len(self.amfList)):
                if self.amfList[i].close == True:
                    log.logger.debug('AMF (%d) has been closed' % (self.amfList[i].id))
                log.logger.debug('AMF (%d) has (%d) messages' % (self.amfList[i].id, self.amfList[i].n_msgs))
        return reward





