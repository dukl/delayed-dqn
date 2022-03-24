import numpy as np
from environment import ENV
from agent_entity import Agent
from stateModel import SM
from actionModel import AM
import matplotlib.pyplot as plt
from logger import log
import talib
from scipy import stats


delta_t = -1 # time step
NUM_UE_REQs = 10
MAX_TIME = 100

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

if __name__ == '__main__':
    env = ENV()
    agent = Agent()
    log.logger.debug('[System][initial the Environment]')
    log.logger.debug('[System][initial the Agent]')
    UeReqs = stats.poisson.rvs(mu=NUM_UE_REQs, size=MAX_TIME + 10, random_state=None)
    print(UeReqs)
    while delta_t<MAX_TIME:
        delta_t += 1
        env.model.update_ue_reqs_every_time_step(UeReqs[delta_t])
        log.logger.debug('[System][time point: %d] begin, adding new %d UE Requests' % (delta_t, UeReqs[delta_t]))
        action = agent.receive_observation(check_state(), delta_t)
        if action != None:
            action_on_road.append(AM(action,delta_t))
        state, reward = env.send_observation(check_action(delta_t), delta_t, UeReqs[delta_t + 1])
        state_on_road.append(SM(state, delta_t, reward))
        if reward.value == None:
            log.logger.debug('[ENV][newly obs: '+''.join(str(state))+']')
            log.logger.debug('[ENV][No reward]')
        else:
            log.logger.debug('[ENV][newly obs: '+''.join(str(state))+']')
            log.logger.debug('[ENV][newly reward: %f]' % (reward.value))
        log.logger.debug('[System][time point: %d end]' % (delta_t))
        #if (delta_t + 1) % 30 == 0:
            #save_plot(delta_t, env.model.amfList)
    save_plot(NUM_UE_REQs, env.model.amfList)



