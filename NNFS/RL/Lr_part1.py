import os
import gym
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

environment_name = 'CartPole-v0'
#env = gym.make(environment_name)

log_path = os.path.join('Training', 'Logs')
PPO_Path = os.path.join('Training', 'Saved Models', 'PPO_Model_Cartpole')

# Instantiate
env = gym.make(environment_name)
env = DummyVecEnv([lambda: env])
#model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_path)

#model.save(PPO_Path)
#del model


'''
model = PPO.load(PPO_Path, env=env)

evaluate_policy(model, env, n_eval_episodes=10, render=True)

env.close()

episodes = 5
for episode in range(1, episodes+1):
    obs = env.reset()
    done = False
    score = 0

    while not done:
        env.render()
        action, _ = model.predict(obs) # Now using model here
        obs, reward, done, info = env.step(action)
        score += reward
    print('Episode:{} Score:{}'.format(episode, score))
env.close()
'''
training_log_path = os.path.join(log_path, 'PPO_2')

# -----------------------------------
save_path = os.path.join('Training', 'Saved Models')

stop_callback = StopTrainingOnRewardThreshold(reward_threshold=200, verbose=1)
eval_callback = EvalCallback(env,
                             callback_on_new_best=stop_callback,
                             eval_freq=10000,
                             best_model_save_path=save_path,
                             verbose=1)
model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_path)

model.learn(total_timesteps=20000, callback=eval_callback)


# Defining a new neural network policy
net_arch = [dict(pi=[128,128,128,128], vf=[128,128,128,128])]
model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=log_path, policy_kwargs={'net_arch':net_arch})

model = DQN('MlpPolicy', env, verbose=1, tensorboard_log=log_path)