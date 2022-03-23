import numpy as np
import tensorflow as tf
from environment import ENV
from agent_entity import Agent
from stateModel import SM
from actionModel import AM
from stateModel import log
from logger import log

delta_t = -1 # time step
NUM_UE_REQs = 200

state_on_road = []
action_on_road = []
reward_on_road = []

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

if __name__ == '__main__':
    env = ENV(NUM_UE_REQs)
    agent = Agent()
    log.logger.debug('[System][initial the Environment]')
    log.logger.debug('[System][initial the Agent]')
    while delta_t<=10:
        delta_t += 1
        log.logger.debug('[System][time point: %d] begin' % (delta_t))
        action = agent.receive_observation(check_state(), delta_t)
        if action != None:
            action_on_road.append(AM(action,delta_t))
        state, reward = env.send_observation(check_action(delta_t), delta_t)
        state_on_road.append(SM(state, delta_t))
        log.logger.debug('[System][time point: %d end]\n' % (delta_t))


