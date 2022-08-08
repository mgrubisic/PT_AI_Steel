import os
import numpy as np
import pickle

from gym import Env
from gym.spaces import Discrete, MultiDiscrete
from gym.utils.env_checker import check_env
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from NNFS.RL.steelProfilesRL import getProfileList, checkStrongProfile, checkUtil, evaluateProfile

# Build an agent to give us the best beam for the problem
# Goal: return the beam with the lowest environmental impact

# --- Flags

class BeamEnv(Env):
    def __init__(self, maxUtil, IPE=False, HEA=False, HEB=False, HEM=False, KVHUP=False, REKHUP=False):

        # Initializing system parameters
        self.typeProfiles = [IPE, HEA, HEB, HEM, KVHUP, REKHUP]
        self.profiles = getProfileList(IPE=IPE, HEA=HEA, HEB=HEB, HEM=HEM, KVHUP=KVHUP, REKHUP=REKHUP)
        self.profiles.sort(key=lambda x: x.getWeight() * x.getEPD())
        n_profiles = len(self.profiles)

        # Defining environment parameters
        self.action_space = Discrete(n_profiles)
        self.observation_space = MultiDiscrete([100, 150, 150])

        #self.observation = self.observation_space.sample()
        self.done = False


        self.bestCO2_standard = self.profiles[-1].getCO2()
        self.bestCO2 = self.bestCO2_standard
        self.maxUtil = maxUtil
        self.standard_episode_length = 50
        self.episode_length = self.standard_episode_length
        self.penalty = 10
        self.penaltyCO2 = 10

        # trackers
        self.actionList = np.zeros(n_profiles)
        self.stateList = np.zeros([4, 1])


    def step(self, action):
        # action is a discrete value
        # the observation space is a [a, b, c] vector
        # where a, b, c = length, dead load, live load
        self.episode_length -= 1
        util = checkUtil(self.profiles[action], self.observation[0] / 10,
                         self.observation[1] / 10, self.observation[2] / 10)

        # reward scheme
        if util > self.maxUtil:
            reward = -1 * util/self.maxUtil * self.penalty
        else:
            CO2ekv = self.profiles[action].getCO2()
            if self.bestCO2 > CO2ekv:
                reward = self.bestCO2 - CO2ekv
                self.bestCO2 = CO2ekv
            else:
                reward = -(CO2ekv - self.bestCO2) / CO2ekv * self.penaltyCO2

        # return and check for done
        info={}
        if self.episode_length <= 0:
            self.done = True
            self.actionList[action] += 1
            self.stateList = np.append(self.stateList, [[action],
                                                         [self.observation[0] / 10],
                                                         [self.observation[1] / 10],
                                                         [self.observation[2] / 10]], axis=1)
        else:
            self.done = False

        return self.observation, reward, self.done, info

    def render(self):
        pass

    def getLists(self):
        return self.actionList, self.stateList

    def reset(self):
        checkPossible = False
        while not checkPossible:
            self.observation = self.observation_space.sample()
            checkPossible, _ = self.checkIfPossible(self.observation, self.typeProfiles)


        self.episode_length = self.standard_episode_length
        self.done = False
        self.bestCO2 = self.bestCO2_standard
        return self.observation

    def checkIfPossible(self, obs, typeProfiles):
        return checkStrongProfile(obs[0] / 10, obs[1] / 10, obs[2] / 10, [i for i in typeProfiles])


# PARAMETERS
if __name__ == '__main__':

    maxUtil = 1.0
    EPD_valset = 1.18 # kg CO2-ekv / kg steel. Navn: NEPD-2526-1260-NO Bjelker og formstål
    EPD_ColdFo = 2.49 # kg CO2-ekv / kg steel. Navn: NEPD-2525-1263-NO Kaldformet hulprofil
    EPD = [EPD_valset, EPD_ColdFo]
    #
    env = BeamEnv(maxUtil, IPE=True, HEA=True, HEB=True, HEM=True, KVHUP=True, REKHUP=True)

    log_path = os.path.join('TrainingBeam', 'Logs')
    PPO_Path = os.path.join('TrainingBeam', 'SavedModels', 'FKON')
    training_log_path = os.path.join(log_path, 'SavedModels')
    save_path = os.path.join('TrainingBeam', 'SavedModels')
    saved_file = os.path.join('TrainingBeam', 'SavedModels', 'best_model.zip')

    stop_callback = StopTrainingOnRewardThreshold(reward_threshold=900, verbose=1)
    eval_callback = EvalCallback(env,
                                 callback_on_new_best=stop_callback,
                                 eval_freq=200,
                                 best_model_save_path=save_path,
                                 verbose=1)
    model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_path, device='cpu')
    #model = PPO.load(saved_file, env=env, tensorboard_log=log_path)

    model.learn(total_timesteps=1000000, callback=eval_callback)
    #model.save(saved_file)
    actionList, stateList = env.getLists()
    pickle.dump(actionList, open("actionList.p", "wb"))
    pickle.dump(stateList, open("stateList.p", "wb"))

    env.close()