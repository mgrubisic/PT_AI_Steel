import os

from gym import Env
from gym.spaces import Discrete, Box, Dict, Tuple, MultiBinary, MultiDiscrete
from gym.utils.env_checker import check_env
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from NNFS.RL.steelProfilesRL import getProfileList, checkStrongProfile, checkUtil, evaluateProfile

# Build an agent to give us the best beam for the problem
# Goal: return the beam with lowest environmental impact

# --- Flags

class BeamEnv(Env):
    def __init__(self, maxUtil, EPD, IPE=False, HEA=False, HEB=False, HEM=False, KVHUP=False, REKHUP=False):
        # Initializing system parameters
        self.typeProfiles = [IPE, HEA, HEB, HEM, KVHUP, REKHUP]
        self.profiles = getProfileList(IPE=IPE, HEA=HEA, HEB=HEB, HEM=HEM, KVHUP=KVHUP, REKHUP=REKHUP)
        n_profiles = len(self.profiles)
        self.EPD = EPD
        # Defining environment parameters
        self.action_space = Discrete(n_profiles)
        self.observation_space = Tuple((Discrete(12), Box(1, 15, shape=(2,))))

        if self.profiles[n_profiles - 1].isIprofile:
            EPD = EPD[0]
        else:
            EPD = EPD[1]

        self.bestCO2_standard = self.profiles[n_profiles - 1].weight * EPD
        self.bestCO2 = self.bestCO2_standard
        self.maxUtil = maxUtil
        self.standard_episode_length = 20
        self.episode_length = self.standard_episode_length
        self.penalty = 100

    def step(self, action):
        # action is a discrete value
        # the observation space is a [a, [b, c]] vector
        # where a, b, c = length, dead load, live load

        util = checkUtil(self.profiles[action], self.observation[0], self.observation[1][0], self.observation[1][1])

        reward = 0
        if util > self.maxUtil:
            reward = -util/self.maxUtil * self.penalty
        else:
            EPD = EPD_ColdFo if self.profiles[action].isIprofile else EPD_valset

            weight = evaluateProfile(self.profiles[action], self.observation[0], self.observation[1][0], self.observation[1][1])
            CO2ekv = weight * EPD
            if self.bestCO2 > CO2ekv:
                reward += self.bestCO2 - CO2ekv
                self.bestCO2 = CO2ekv
            else:
                reward = -0.5


        info={}
        if self.episode_length <=0:
            self.done = True
        else:
            self.done = False

        return self.observation, reward, self.done, info

    def render(self):
        pass

    def reset(self):
        checkPossible = False
        while not checkPossible:
            self.observation = self.observation_space.sample()
            checkPossible, worstCO2 = self.checkIfPossible(self.observation, self.typeProfiles, self.EPD)

        self.episode_length = self.standard_episode_length
        self.done = False
        self.bestCO2 = worstCO2
        return self.observation

    def checkIfPossible(self, obs, typeProfiles, EPD):
        return checkStrongProfile(obs[0], obs[1][0], obs[1][1], [i for i in typeProfiles], EPD)


# PARAMETERS
if __name__ == '__main__':

    maxUtil = 1.0
    EPD_valset = 1.18 # kg CO2-ekv / kg steel. Navn: NEPD-2526-1260-NO Bjelker og formstål
    EPD_ColdFo = 2.49 # kg CO2-ekv / kg steel. Navn: NEPD-2525-1263-NO Kaldformet hulprofil
    EPD = [EPD_valset, EPD_ColdFo]
    #
    env = BeamEnv(maxUtil, EPD, IPE=True)
    log_path = os.path.join('TrainingBeam', 'Logs')
    PPO_Path = os.path.join('TrainingBeam', 'SavedModels', 'FKON')
    training_log_path = os.path.join(log_path, 'SavedModels')
    save_path = os.path.join('TrainingBeam', 'SavedModels')
    saved_file = os.path.join('TrainingBeam', 'SavedModels', 'best_model.zip')

    stop_callback = StopTrainingOnRewardThreshold(reward_threshold=900, verbose=1)
    eval_callback = EvalCallback(env,
                             callback_on_new_best=stop_callback,
                             eval_freq=100,
                             best_model_save_path=save_path,
                             verbose=1)
    model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_path)
    #model = PPO.load(saved_file, env=env, tensorboard_log=log_path)

    model.learn(total_timesteps=3000, callback=eval_callback)
    model.save(saved_file)

    env.close()