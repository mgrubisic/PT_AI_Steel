
import time
import os

from waiting import wait, TimeoutExpired
import numpy as np

import watchdog
import watchdog.events
import watchdog.observers
from gym import Env
from gym.spaces import Discrete, Box, Dict, Tuple, MultiBinary, MultiDiscrete
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from NNFS.RL.steelProfilesRL import getProfileList, writeToFocus, readFromFocus

# Build an agent to give us the best truss for the problem.
# Goal: optimizing a truss beam with respect to weight or GWP (global warming potential)

# ---- Flags


FLAG_test = True
# Custom environment
class TrussEnv(Env):
    def __init__(self, maxUtil):
        # Initializing the systems parameters
        self.IPE = False; self.HEB = False; self.HEM = True
        profiles = getProfileList(IPE=self.IPE, HEB=self.HEB, HEM=self.HEM)

        n_profiles = len(profiles)
        # Defining the environment parameters
        self.action_space = MultiDiscrete([n_profiles, n_profiles, n_profiles])
        self.observation_space = Box(200,80000, shape=(1,))

        writeToFocus(n_profiles-1, n_profiles-1, n_profiles-1, IPE=self.IPE, HEB=self.HEB, HEM=self.HEM)
        time.sleep(0.5)
        output = readFromFocus()
        self.state = output[0] # ??????????????
        print(self.state)
        self.bestWeight = 800000.00
        self.episode_length = 20
        self.maxUtil = maxUtil

    def step(self, action):
        global signal
        signal = False

        writeToFocus(action[0], action[1], action[2], IPE=self.IPE, HEB=self.HEB, HEM=self.HEM)

        try:
            wait(lambda: isReadReady(), sleep_seconds=0.01, timeout_seconds=10, waiting_for="something to be ready")
        except TimeoutExpired:
            self.episode_length = 0
            info= {"Timeout": True}
            raise SystemExit(0)
        else:
            pass


        # leser action
        output = [float(x) for x in readFromFocus()]
        self.state = np.array([output[0]]) # readFromFocus[0]
        self.episode_length -= 1

        # Calculate reward. Set up reward scheme
        if max(output[2:]) > self.maxUtil:
            reward = -1
        elif self.state[0] < self.bestWeight:
            reward = (self.bestWeight - self.state[0]) / 10
            self.bestWeight = self.state[0]

        elif self.state[0] == self.bestWeight:
            reward = 0
        else:
            reward = -1
            # reward = self.bestWeight - self.state[0]

        if self.episode_length <=0:
            self.done=True
        else:
            self.done=False


        info={}
        return self.state, reward, self.done, info

    def render(self):
        pass
        '''plt.scatter(*zip(*self.upper_nodes))
        plt.plot(*zip(*self.upper_nodes))
        plt.scatter(*zip(*self.lower_nodes))
        plt.plot(*zip(*self.lower_nodes))
        plt.plot(*zip(*self.all_nodes))
        plt.show()'''
    def reset(self):
        self.episode_length = 20
        self.state = np.array([80000])
        self.bestWeight = 80000
        self.done = False
        print("I am reset")
        return self.state


    #def calculateTruss(self, action):
    #    return random.randint(200, 800000)



class Handler(watchdog.events.PatternMatchingEventHandler):
    def __init__(self):
        # Set the patterns for PatternMatchingEventHandler
        watchdog.events.PatternMatchingEventHandler.__init__(self, patterns=['Example.txt'],
                                                             ignore_directories=True, case_sensitive=False)

    def on_modified(self, event):
        #print("Modified - % s." % event.src_path)

        global signal
        signal = True


def isReadReady():
    if signal:
        return True
    return False


#def episode(antall):
    # følg "clean code prinsipp" eller no. Ting skal bare ha en utvei.
    #

if __name__ == '__main__':
    global signal
    src_path = "../../../../../../output/"
    event_handler = Handler()
    observer = watchdog.observers.Observer()
    observer.schedule(event_handler, path=src_path, recursive=False)
    observer.start()


    # -------- PARAMETERS
    maxUtil = 1.0



    # -------- START
    env = TrussEnv(maxUtil)
    env.reset()
    '''episodes = 2
    for episode in range(1, episodes + 1):
        obs = env.reset()
        done=False
        score = 0

        while not done:
            # env.render()
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            score += reward
        print('Episode:{} Score:{}'.format(episode, score))
    '''


    #check_env(env)

    log_path = os.path.join('Training', 'Logs')
    PPO_Path = os.path.join('Training', 'SavedModels', 'FKON')
    training_log_path = os.path.join(log_path, 'SavedModels')
    save_path = os.path.join('Training', 'SavedModels')
    saved_file = os.path.join('Training', 'SavedModels', 'best_model.zip')

    stop_callback = StopTrainingOnRewardThreshold(reward_threshold=90000, verbose=1)
    eval_callback = EvalCallback(env,
                             callback_on_new_best=stop_callback,
                             eval_freq=50,
                             best_model_save_path=save_path,
                             verbose=1)
    model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_path)
    model = PPO.load(saved_file, env=env, tensorboard_log=log_path)

    model.learn(total_timesteps=3000, callback=eval_callback)
    model.save(saved_file)

    observer.stop()
    env.close()