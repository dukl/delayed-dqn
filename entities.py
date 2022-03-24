from logger import log
from queue import Queue
reqMsgs = ['RegistrationRequest', 'AuthenticationRequst', 'AuthenticationResponse', 'SecurityModeCommand', 'SecurityModeComplete', 'RegistrationComplete', 'NULL']

class AmfEntity():
    def __init__(self, n_cpu_cores, id, tp):
        self.n_called = -1
        self.tp = tp
        self.n_cpu_cores = n_cpu_cores
        self.n_msgs = 0
        self.n_iterate = 0
        self.n_msgs_record = []
        self.time_point = []
        self.id = id
        self.close = False
        self.sameMsgIsProcessing = False
        self.message_queue = Queue(maxsize=0)
        self.oldMsg = Msgs(0, 0, 'NULL')
        self.newMsg = Msgs(0, 0, 'NULL')
    def stateTrans(self, ue_id, msg_id, msgType, dt):
        self.n_called += 1
        #log.logger.debug('AMF (%d) received Message (%d, %d, %s)' % (self.id, ue_id, msg_id, msgType))
        if msg_id > 0:
            message = Msgs(ue_id, msg_id, msgType)
            self.message_queue.put(message)
        if self.message_queue.qsize() == 0 and self.sameMsgIsProcessing == False:
            self.n_msgs_record.append(0)
            self.time_point.append(self.tp + self.n_called*dt)
            #log.logger.debug('No Msg into AMF (%d) ...' % (self.id))
            if self.close == True:
                log.logger.debug('AMF[%d] has been closed' % (self.id))
            self.newMsg.ue_id = 0
            self.newMsg.msg_id = 0
            self.newMsg.msgType = 'NULL'
            #log.logger.debug('AMF (%d) remains (%d) message' % (self.id, self.message_queue.qsize()))
            #log.logger.debug('AMF (%d) return message (%d, %d, %s)' % (self.id, self.newMsg.ue_id, self.newMsg.msg_id, self.newMsg.msgType))
            return self.newMsg
        if self.sameMsgIsProcessing == True:
            #log.logger.debug('AMF (%d) iterate (%d)' % (self.id, self.n_iterate))
            #log.logger.debug('AMF (%d): Message Old (%d, %d, %s) has been processed (%f)' % (self.id, self.oldMsg.ue_id, self.oldMsg.msg_id, self.oldMsg.msgType, self.n_cpu_cores * 5 * dt * self.n_iterate))
            if self.n_cpu_cores*5*dt*self.n_iterate >= 1:
                self.n_msgs = self.message_queue.qsize()
                self.sameMsgIsProcessing = False
                self.n_iterate = 0
                if self.oldMsg.msg_id + 1 > 6:
                    #log.logger.debug('AMF (%d) doesnot generate new message'%(self.id))
                    self.newMsg.ue_id = 0
                    self.newMsg.msg_id = 0
                    self.newMsg.msgType = 'NULL'
                else:
                    self.newMsg.ue_id = self.oldMsg.ue_id
                    self.newMsg.msg_id = self.oldMsg.msg_id + 1
                    self.newMsg.msgType = reqMsgs[self.oldMsg.msg_id]
                    #log.logger.debug('AMF (%d) generates new message (%d, %d, %s)'%(self.id, self.newMsg.ue_id, self.newMsg.msg_id, self.newMsg.msgType))
            else:
                self.n_iterate += 1
                self.n_msgs = self.message_queue.qsize() + 1
        elif self.message_queue.qsize() > 0 and self.sameMsgIsProcessing == False:
            message = self.message_queue.get()
            #log.logger.debug('AMF (%d) is processing Message (%d, %d, %s)' % (self.id, message.ue_id, message.msg_id, message.msgType))
            self.sameMsgIsProcessing = True
            self.oldMsg = message
            #log.logger.debug('AMF (%d) iterate (%d)' % (self.id, self.n_iterate))
            #log.logger.debug('AMF (%d): Message New (%d, %d, %s) has been processed (%f)'%(self.id, message.ue_id, message.msg_id, message.msgType, self.n_cpu_cores*5*dt*self.n_iterate))
            self.n_iterate += 1
            self.newMsg.ue_id = 0
            self.newMsg.msg_id = 0
            self.newMsg.msgType = 'NULL'
            self.n_msgs = self.message_queue.qsize() + 1
        self.n_msgs_record.append(self.n_msgs)
        self.time_point.append(self.tp + self.n_called * dt)
        #log.logger.debug('AMF (%d) remains (%d) message' % (self.id, self.n_msgs))
        #log.logger.debug('AMF (%d) return message (%d, %d, %s)' % (self.id, self.newMsg.ue_id, self.newMsg.msg_id, self.newMsg.msgType))
        return self.newMsg

class Msgs():
    def __init__(self, ue_id, msg_id, msgType):
        self.ue_id = ue_id
        self.msgType = msgType
        self.msg_id = msg_id


