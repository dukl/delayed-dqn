import copy

import numpy as np
from environment import ENV
from agent_entity import Agent
from stateModel import SM
from actionModel import AM
import matplotlib.pyplot as plt
from logger import log, logR
import talib
from scipy import stats
from entities import AmfEntity


delta_t = -1 # time step
NUM_UE_REQs = 20
MAX_TIME = 1000

state_on_road = []
action_on_road = []

def check_action(delta_t):
    acts_in_env = []
    index = []
    msg = '[leaving acts -->'
    for i in range(len(action_on_road)):
        msg += 'a[' + str(action_on_road[i].id) + '] '
        if action_on_road[i].check_status(delta_t) == 'Arrived':
            index.append(i)
    log.logger.debug('[System]' + msg + ']')
    log.logger.debug('[System][checking acts status (index)--> ' + ''.join(str(index)) + ']')
    for i in range(len(index)):
        acts_in_env.append(action_on_road[index[len(index) - i - 1]])
        del action_on_road[index[len(index) - i - 1]]
    index.clear()
    log.logger.debug('[System][size of action_on_road: %d]' % (len(action_on_road)))
    return acts_in_env

def check_state():
    obs_in_agent = []
    index = []
    msg = '[leaving obs --> '
    for i in range(len(state_on_road)):
        msg += 's['+str(state_on_road[i].id)+'] '
        if state_on_road[i].check_status() == 'Arrived':
            index.append(i)
    log.logger.debug('[System]'+msg+']')
    log.logger.debug('[System][checking obs status (index)--> '+''.join(str(index))+']')
    for i in range(len(index)):
        obs_in_agent.append(state_on_road[index[len(index)-i-1]])
        del state_on_road[index[len(index)-i-1]]
    index.clear()
    log.logger.debug('[System][size of state_on_road: %d]' % (len(state_on_road)))
    return obs_in_agent

def save_plot(num, amfList):
    fig, ax = plt.subplots()
    #color = ['red', 'blue', 'black', 'green', 'pink', 'orange', 'violet', 'lawngreen', 'dodgerblue', 'magenta']
    for i in range(len(amfList)):
        sma = talib.SMA(np.array(amfList[i].n_msgs_record).astype(float), timeperiod=10)
        ax.plot(np.array(amfList[i].time_point), sma)
    plt.savefig(str(num)+'.jpg')
    plt.show()

def envCopy(env):
    log.logger.debug('[Original ENV][msgInRISE][%d][amflist: %d]' % (len(env.model.msgInRISE), len(env.model.amfList)))
    copied_env = ENV(True)
    log.logger.debug('[-DeepCopied ENV][msgInRISE][%d][amflist: %d]' % (len(copied_env.model.msgInRISE), len(copied_env.model.amfList)))
    copied_env.model.total_ue_reqs = env.model.total_ue_reqs
    copied_env.model.n_ue_reqs = env.model.n_ue_reqs
    copied_env.model.n_amf_insts = env.model.n_amf_insts
    copied_env.model.it_time = env.model.it_time
    copied_env.model.time_interval = env.model.time_interval
    copied_env.model.Delay_Up_Link = env.model.Delay_Up_Link
    copied_env.model.Delay_Down_Link = env.model.Delay_Down_Link
    for i in range(len(env.model.msgUpOnRoad)):
        copied_env.model.msgUpOnRoad.append(copy.deepcopy(env.model.msgUpOnRoad[i]))
    for i in range(len(env.model.msgDnOnRoad)):
        copied_env.model.msgDnOnRoad.append(copy.deepcopy(env.model.msgDnOnRoad[i]))
    for i in range(len(env.model.msgInRISE)):
        copied_env.model.msgInRISE.append(copy.deepcopy(env.model.msgInRISE[i]))
    for i in range(len(env.model.inputMsgs)):
        copied_env.model.inputMsgs.append(env.model.inputMsgs[i])
    for i in range(len(env.model.flag)):
        copied_env.model.flag.append(env.model.flag[i])
    copied_env.model.AMFIndex = env.model.AMFIndex
    for i in range(len(env.model.amfList)):
        copied_env.model.amfList.append(AmfEntity(env.model.amfList[i].n_cpu_cores, env.model.amfList[i].id, env.model.amfList[i].tp))
        copied_env.model.amfList[-1].n_msgs_record = copy.deepcopy(env.model.amfList[i].n_msgs_record)
        copied_env.model.amfList[-1].time_point = copy.deepcopy(env.model.amfList[i].time_point)
        for j in range(len(env.model.amfList[i].message_queue)):
            copied_env.model.amfList[-1].message_queue.append(copy.deepcopy(env.model.amfList[i].message_queue[j]))

    copied_env.model.numAMF = env.model.numAMF
    for i in range(len(env.model.cpuInAMF)):
        copied_env.model.cpuInAMF.append(copy.deepcopy(env.model.cpuInAMF[i]))
    copied_env.model.add_new_action = env.model.add_new_action
    copied_env.model.MAX_AMF_INST = env.model.MAX_AMF_INST
    copied_env.model.usefulUpRoad = env.model.usefulUpRoad
    copied_env.model.amf_id = env.model.amf_id
    log.logger.debug('[DeepCopied ENV][msgInRISE][%d][amflist: %d]' % (len(copied_env.model.msgInRISE), len(copied_env.model.amfList)))

    return copied_env


if __name__ == '__main__':
    env = ENV()
    agent = Agent()
    log.logger.debug('[System][initial the Environment]')
    log.logger.debug('[System][initial the Agent]')
    learn_index = 0
    load = []
    for ep in range(1000):
        load.clear()
        log.logger.debug('[System][Episode][%d]' % (ep+1))
        env.reset()
        agent.reset()
        agent.reward_sum = 0
        UeReqs = stats.poisson.rvs(mu=NUM_UE_REQs, size=MAX_TIME + 10, random_state=None)
        log.logger.debug('[System][Input Msgs] \n %s' % (str(UeReqs)))
        delta_t = -1
        state_on_road.clear()
        action_on_road.clear()
        while delta_t<MAX_TIME:
            loadV = UeReqs[delta_t]+len(env.model.msgInRISE)+env.model.usefulUpRoad+len(env.model.msgDnOnRoad)
            for i in range(len(env.model.amfList)):
                loadV += env.model.amfList[i].n_msgs
            load.append(loadV)
            delta_t += 1
            env.model.update_ue_reqs_every_time_step(UeReqs[delta_t])
            log.logger.debug('[System][time point: %d] begin, adding new %d UE Requests' % (delta_t, UeReqs[delta_t]))
            action, delay_a, old_obs_id = agent.receive_observation(check_state(), delta_t)
            if action != None:
                action_on_road.append(AM(action,delta_t, delay_a, old_obs_id))
            if env.model.isWithDelay is True:
                state, reward = env.send_observation(check_action(delta_t), delta_t, UeReqs[delta_t])
            else:
                state, reward = env.send_observation_no_delay(check_action(delta_t), delta_t, UeReqs[delta_t])
            if state is None:
                log.logger.debug('[ENV][---- Not received action, donnot collect state ----]\n')
                continue
            state_on_road.append(SM(state, delta_t, reward))
            state_on_road[-1].env = envCopy(env)
            state_on_road[-1].inputMsgs = 0
            state_on_road[-1].env.model.inputMsgs.append(UeReqs[delta_t + 1])
            #state_on_road[-1].env.model.inputMsgs.append(UeReqs[delta_t + 2])
            #state_on_road[-1].env.model.inputMsgs.append(UeReqs[delta_t + 3])
            state_on_road[-1].env.model.flag.append(False)
            #state_on_road[-1].env.model.flag.append(False)
            #state_on_road[-1].env.model.flag.append(False)
            log.logger.debug('env input msgs (%d)/ flag (%d)' % (len(env.model.inputMsgs), len(env.model.flag)))
            log.logger.debug('envGT input msgs (%d)/ flag (%d)' % (len(state_on_road[-1].env.model.inputMsgs), len(state_on_road[-1].env.model.flag)))

            if reward.value == 0:
                log.logger.debug('[ENV][newly obs: '+''.join(str(state))+']')
                log.logger.debug('[ENV][No reward]')
            else:
                log.logger.debug('[ENV][newly obs: '+''.join(str(state))+']')
                log.logger.debug('[ENV][newly reward: %f with action %d]' % (reward.value, env.action_seq[-1]))
            log.logger.debug('[System][time point: %d end]\n' % (delta_t))
        logR.logger.debug('Epision[%d] DReward %f' % (ep, agent.reward_sum))
        logR.logger.debug('Episode discard rate: (%d / %d = %f)' % (env.model.n_discard_msgs, env.model.n_request_msgs, env.model.n_discard_msgs/env.model.n_request_msgs))
        agent.epison_reward.append(agent.reward_sum)
        logR.logger.debug('action sequences \n%s' % (str(env.action_seq)))
        #if (delta_t + 1) % 30 == 0:
            #save_plot(delta_t, env.model.amfList)
    #save_plot(NUM_UE_REQs, env.model.amfList)
    acts = []
    acts.append(1)
    acts.append(1)
    for i in range(len(env.action_seq)):
        tmp = acts[-1] + env.action_seq[i]
        if tmp == 0 or tmp > 10:
            acts.append(acts[-1])
            acts.append(acts[-1])
        else:
            acts.append(tmp)
            acts.append(tmp)

    fig, ax = plt.subplots()
    ax.plot(load, color='green', label='Loads')
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Number of Unprocessed Signaling Messages')
    ax.legend(loc=0)
    ax2 = ax.twinx()
    x = [j for j in range(len(acts))]
    ax2.step(x, acts, color='blue',label='AMF Instances')
    ax2.set_ylabel('Number of AMF Instances')
    ax2.legend(loc=0)
    plt.savefig('acts-load.png',bbox_inches='tight',dpi=fig.dpi,pad_inches=0.1)
    #plt.show()
    #plt.plot(agent.epison_reward)
    #plt.savefig('rewards-0504-nd.png')
    #plt.show()



