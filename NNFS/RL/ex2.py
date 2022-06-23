import watchdog
import time
import sys
from watchdog import events, observers
from NNFS.RL.steelProfilesRL import getProfileList, readFromFocus, writeToFocus
from gym import Env
from gym.spaces import Box, MultiDiscrete

global episode, episodes, episode_length, i_step, done, score

class TailRecurseException(BaseException):
  def __init__(self, args, kwargs):
    self.args = args
    self.kwargs = kwargs

def tail_call_optimized(g):
    def func(*args, **kwargs):
        f = sys._getframe()
        if f.f_back and f.f_back.f_back \
                and f.f_back.f_back.f_code == f.f_code:
            raise TailRecurseException(args, kwargs)
        else:
            while 1:
                try:
                    return g(*args, **kwargs)
                except TailRecurseException as e:
                    args = e.args
                    kwargs = e.kwargs

    func.__doc__ = g.__doc__
    return func

class Handler(watchdog.events.PatternMatchingEventHandler):
    def __init__(self):
        # Set the patterns for PatternMatchingEventHandler
        watchdog.events.PatternMatchingEventHandler.__init__(self, patterns=['Example.txt'],
                                                             ignore_directories=True, case_sensitive=False)

    def on_modified(self, event):
        pass


class TrEnv(Env):
    def __init__(self):
        # Initialising the system
        profiles = getProfileList(IPE=True, HEA=True, HEM=True)
        n_profiles = len(profiles)

        # Defining the environment parameters
        self.action_space = MultiDiscrete([n_profiles, n_profiles, n_profiles])
        self.observation_space = Box(200, 80000, shape=(1,))
        writeToFocus(n_profiles - 1, n_profiles - 1, n_profiles - 1)
        time.sleep(0.4)
        output = readFromFocus()
        self.state = output[0]
        self.firstResult = output[0]
        self.bestWeight = self.state
        self.episodes = episodes

    def step(self, action):
        pass
        reward = 0
        done = False

        info = {}
        return self.state, reward, done, info

    def render(self):
        pass

    def reset(self):
        self.state = self.firstResult
        return self.state


def runEpisode():
    global episode, done, score, i_step
    episode += 1
    print("Run episode, episode:" + str(episode))
    i_step = 0

    env.reset()
    done = False
    score = 0

    runStep()
    pass

def runStep():
    global i_step
    i_step += 1
    print("Run step, step:" + str(i_step))

    endStep()

def endStep():
    global i_step, episode_length

    if i_step < episode_length:
        runStep()
    else:
        endEpisode()


def endEpisode():
    global episode, episodes
    print("End episode, episode:" + str(episode))

    # Reset stuff
    if episode < episodes:
        runEpisode()
    else:
        endAll()


def endAll():
    print("End all")


if __name__ == '__main__':

    episode_length = 98
    episodes = 5
    env = TrEnv()

    src_path = "../../../../../../output/"
    event_handler = Handler()
    observer = watchdog.observers.Observer()
    observer.schedule(event_handler, path=src_path, recursive=False)
    observer.start()

    print("main")
    episode = 0
    i_step = 0
    runEpisode()