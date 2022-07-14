from gym import Env
from gym.spaces import Discrete, Box, Dict, Tuple, MultiBinary, MultiDiscrete
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

        # Defining environment parameters
        self.action_space = Discrete(n_profiles)
        self.observation_space = Tuple((Discrete(12), Box(1, 15, shape=(2,))))

        if self.profiles[n_profiles - 1].isIprofile:
            self.EPD = EPD[0]
        else:
            self.EPD = EPD[1]
        self.observation = self.profiles[n_profiles - 1].weight * self.EPD
        self.bestWeight = self.observation
        self.maxUtil = maxUtil
        self.standard_episode_length = 20
        self.episode_length = self.standard_episode_length

    def step(self, action):
        # action is a discrete value
        # the observation space is a [a, [b, c]] vector
        # where a, b, c = length, dead load, live load

        util = checkUtil(self.profiles(action), self.observation_space[0], self.observation_space[1][0], self.observation_space[1][1])


        if util > self.maxUtil:
            reward = -util/self.maxUtil
        else:
            EPD = EPD_ColdFo if self.profiles(action).isIprofiles else EPD_valset

            weight = evaluateProfile(self.profiles(action), self.observation_space[0], self.observation_space[1][0], self.observation_space[1][1])
            CO2ekv = weight * EPD
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
            checkPossible = self.checkIfPossible(self.observation, self.typeProfiles)

        self.episode_length = self.standard_episode_length
        self.done = False
        return self.observation

    def checkIfPossible(self, obs, typeProfiles):
        return checkStrongProfile(obs[0], obs[1][0], obs[1][1], [i for i in typeProfiles])


# PARAMETERS
maxUtil = 1.0
EPD_valset = 1.18 # kg CO2-ekv / kg steel. Navn: NEPD-2526-1260-NO Bjelker og formstål
EPD_ColdFo = 2.49 # kg CO2-ekv / kg steel. Navn: NEPD-2525-1263-NO Kaldformet hulprofil
EPD = [EPD_valset, EPD_ColdFo]
#
env = BeamEnv(maxUtil, EPD, IPE=True)