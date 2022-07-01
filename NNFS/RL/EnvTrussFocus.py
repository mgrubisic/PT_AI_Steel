import threading
from datetime import datetime
import time
from waiting import wait
import numpy as np
import random
import matplotlib.pyplot as plt
import watchdog
import watchdog.events
import watchdog.observers
from gym import Env
from gym.spaces import Discrete, Box, Dict, Tuple, MultiBinary, MultiDiscrete
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from NNFS.RL.steelProfilesRL import getProfileList, writeToFocus, readFromFocus

# Build an agent to give us the best truss for the problem.
# Goal: optimizing a truss beam with respect to weight or GWP (global warming potential)

# ---- Flags


FLAG_test = True
# Custom environment
class TrussEnv(Env):
    def __init__(self):
        # Initializing the systems parameters
        profiles = getProfileList(IPE=True, HEB=True, HEM=True)

        n_profiles = len(profiles)
        # Defining the environment parameters
        self.action_space = MultiDiscrete([n_profiles, n_profiles, n_profiles])
        self.observation_space = Box(200,80000, shape=(1,))
        writeToFocus(n_profiles-1, n_profiles-1, n_profiles-1)
        time.sleep(0.3)
        output = readFromFocus()
        self.state = output[0]
        print(self.state)
        self.bestWeight = 800000
        self.episode_length = 10

    def step(self, action):
        global signal
        signal = False

        writeToFocus(action[0], action[1], action[2])
        print("Time writeToFocus: " + str(datetime.now().hour) + ":"
              + str(datetime.now().minute) + ":"
              + str(datetime.now().second) + "."
              + str(datetime.now().microsecond))

        wait(lambda: isReadReady(), sleep_seconds=0.01, timeout_seconds=10, waiting_for="something to be ready")

        # leser action
        output=readFromFocus()
        print("Time readFromFocus: " + str(datetime.now().hour) + ":"
              + str(datetime.now().minute) + ":"
              + str(datetime.now().second) + "."
              + str(datetime.now().microsecond))
        #
        self.state = 1 # readFromFocus[0]
        self.episode_length -= 1

        # Calculate reward. Set up reward scheme
        if self.state < self.bestWeight:
            self.bestWeight = self.state
            reward = 1
        elif self.state == 79999:
            reward = -1.1
        else:
            reward = -1

        if self.episode_length <=0:
            done=True
        else:
            done=False

        info={}
        return self.state, reward, done, info

    def render(self):
        pass
        '''plt.scatter(*zip(*self.upper_nodes))
        plt.plot(*zip(*self.upper_nodes))
        plt.scatter(*zip(*self.lower_nodes))
        plt.plot(*zip(*self.lower_nodes))
        plt.plot(*zip(*self.all_nodes))
        plt.show()'''
    def reset(self):
        self.episode_length = 10
        self.state = 80000
        self.bestWeight = 80000
        return self.state


    def calculateTruss(self, action):
        return random.randint(200, 800000)



class Handler(watchdog.events.PatternMatchingEventHandler):
    def __init__(self):
        # Set the patterns for PatternMatchingEventHandler
        watchdog.events.PatternMatchingEventHandler.__init__(self, patterns=['Example.txt'],
                                                             ignore_directories=True, case_sensitive=False)

    def on_modified(self, event):
        #print("Modified - % s." % event.src_path)
        print()
        print("Time event: " + str(datetime.now().hour) + ":"
              + str(datetime.now().minute) + ":"
              + str(datetime.now().second) +"."
              + str(datetime.now().microsecond))
        global signal
        signal = True
        # ---------------------
        # python logget     18:03:54.845429
        # systemet logget   18:03:54.841425 for lastModified
        # systemet logget   18:03:54.842426 for lastAccessed

        # -------------
        # python logget     18:20:42.252380 for writeTo
        # python logget     18:20:42.247375 for readFrom
        # python logget     18:20:42.241370 for event
        # systemet logget   18:20:42.236365 for lastModified


def isReadReady():
    if signal:
        return True
    return False


#def episode(antall):
    # følg "clean code prinsipp" eller no. Ting skal bare ha en utvei.
    #

if __name__ == '__main__':
    global signal # denne trenger vi ikke til slutt
    env = TrussEnv()
    episodes = 5
    src_path = "../../../../../../output/"
    event_handler = Handler()
    observer = watchdog.observers.Observer()
    observer.schedule(event_handler, path=src_path, recursive=False)
    observer.start()
    # def episode(episodes+1)
    startT = time.time()
    for episode in range(1, episodes + 1):
        obs = env.reset()
        done = False
        score = 0

        while not done:
            # env.render()
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            score += reward
        print('Episode:{} Score:{} Time:{}'.format(episode, score, (time.time() - startT)))

    observer.stop()
    env.close()