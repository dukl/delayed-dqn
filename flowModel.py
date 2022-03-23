from queue import Queue
from logger import log
import entities
from entities import Msgs
from entities import AmfEntity
import numpy as np

class FM:
    def __init__(self, n_ue_reqs):
        self.n_ue_reqs = n_ue_reqs
        self.n_amf_insts = 1
        self.it_time = 0
        self.time_interval = 0.01
        self.Delay_Up_Link = 0.5
        self.Delay_Down_Link = 0.5
        self.Rate_SCTP_Out   = 100
        self.msgUpOnRoad = Queue(maxsize=0)
        self.msgDnOnRoad = Queue(maxsize=0)
        self.msgInRISE   = Queue(maxsize=0)
        self.AMFIndex = 0
        self.amfList  = []
        self.numAMF   = 1
        self.cpuInAMF = []
        self.initialize()
        self.add_new_action = False

    def initialize(self):
        for i in range(self.numAMF):
            self.amfList.append(AmfEntity(np.random.uniform(2,4,None), i))
        for i in range(self.n_ue_reqs):
            message = Msgs(i+1, 1, 'RegistrationRequest')
            self.msgInRISE.put(message)

    def check_available_AMF_Inst(self):
        index = self.AMFIndex
        min_n_msgs = 10000
        for i in range(self.numAMF):
            if self.amfList[i].close == True:
                continue
            elif self.amfList[i].n_msgs < min_n_msgs:
                min_n_msgs = self.amfList[i].n_msgs
                self.AMFIndex = i

    def step(self, action, id, tpS, tpE, delta_t):
        if action == None:
            log.logger.debug('[FlowModel][No action is executed]')

        self.add_new_action = False

        while self.it_time < tpE + delta_t - 1:
            if action != None and self.it_time >= tpS + delta_t - 1 and self.add_new_action == False:
                self.add_new_action = True
                log.logger.debug('[FlowModel][Action a[%d] = %d is executed]' % (id, action))
                if action == 1:
                    self.numAMF += 1
                    self.amfList.append(AmfEntity(np.random.uniform(2, 4, None), self.numAMF - 1))
                if action == -1:
                    self.numAMF -= 1
                    if self.numAMF <= 0:
                        log.logger.debug('Number of AMF instance is less than 1, so missed this action, return reward -100')
                    else:
                        self.amfList.sort(key=lambda AmfEntity: AmfEntity.message_queue.qsize(), reverse=False)
                        self.amfList[0].close = True
            self.it_time += self.time_interval
            log.logger.debug('[FlowModel][At time point: %f]' % (self.it_time))
            log.logger.debug('[FlowModel][No. of Msgs in RISE: %d]' % (self.msgInRISE.qsize()))
            if self.msgInRISE.qsize() > 0:
                message = self.msgInRISE.get()
                log.logger.debug('Message (%d, %d, %s) has been out of RISE, waiting to be AMF' % (message.ue_id, message.msg_id, message.msgType))
                self.msgUpOnRoad.put(message)
            else:
                message = Msgs(0,0,'NULL')
                log.logger.debug('No Messages in RISE already')
                self.msgUpOnRoad.put(message)
            if self.msgUpOnRoad.qsize() >= self.Delay_Up_Link/self.time_interval:
                log.logger.debug('msgUpOnRoad is full ...')
                message = self.msgUpOnRoad.get()
                if message.msg_id > 0:
                    self.check_available_AMF_Inst()
                    log.logger.debug('Message (%d, %d, %s) has arieved at AMF (%d), to be processed' % (message.ue_id, message.msg_id, message.msgType, self.AMFIndex))
                    self.amfList[self.AMFIndex].stateTrans(message.ue_id, message.msg_id, message.msgType, self.time_interval)
                    for i in range(self.numAMF):
                        if i == self.AMFIndex:
                            continue
                        else:
                            self.amfList[i].stateTrans(0,0,'NULL',self.time_interval)
                    self.AMFIndex = (self.AMFIndex + 1) % self.numAMF
                else:
                    log.logger.debug('No message is put into AMFs')
                    for i in range(self.numAMF):
                        self.amfList[i].stateTrans(0,0,'NULL', self.time_interval)
            else:
                for i in range(self.numAMF):
                    self.amfList[i].stateTrans(0, 0, 'NULL', self.time_interval)
            for i in range(self.numAMF):
                if self.amfList[i].newMsg.msg_id > 0:
                    log.logger.debug('AMF (%d) generate new message (%d, %d, %s) into msgDnOnRoad (%d)' % (i, self.amfList[i].newMsg.ue_id, self.amfList[i].newMsg.msg_id, self.amfList[i].newMsg.msgType, self.msgDnOnRoad.qsize()))
                    message = Msgs(0,0,'NULL')
                    message.ue_id = self.amfList[i].newMsg.ue_id
                    message.msg_id = self.amfList[i].newMsg.msg_id
                    message.msgType = self.amfList[i].newMsg.msgType
                    self.msgDnOnRoad.put(message)
            if self.msgDnOnRoad.qsize() >= self.Delay_Down_Link / self.time_interval:
                log.logger.debug('msgDnOnRoad is full ...')
                message = self.msgDnOnRoad.get()
                #log.logger.debug(message.ue_id, message.msg_id, message.msgType)
                log.logger.debug('msgDnOnRoad.get: message (%d, %d, %s)' % (message.ue_id, message.msg_id, message.msgType))
                if message.msg_id > 0 and message.msg_id < 6:
                    back_message = Msgs(message.ue_id, message.msg_id + 1, entities.reqMsgs[message.msg_id])
                    log.logger.debug('Message (%d, %d, %s) has been sent back to RISE, RISE generate new message (%d, %d, %s)' % (message.ue_id, message.msg_id, message.msgType, back_message.ue_id, back_message.msg_id, back_message.msgType))
                    self.msgInRISE.put(back_message)
                else:
                    log.logger.debug('RISE dosennot genearte new message any more')
            log.logger.debug('Statics (%d) AMFs ...' % (self.numAMF))
            for i in range(self.numAMF):
                log.logger.debug('AMF (%d) has (%d) messages' % (i, self.amfList[i].n_msgs))






