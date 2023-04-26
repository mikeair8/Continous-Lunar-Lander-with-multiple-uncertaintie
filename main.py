import torch
from ddpg_agent import DDPGAgent
import gym
import Box2D
from gym.wrappers import RecordVideo
from collections import deque
import numpy as np
import moviepy
import random
import csv

env = gym.make("LunarLander-v2", continuous = True, render_mode = 'rgb_array')
env.metadata = {
    'render.modes': ['human', 'rgb_array'],
    'render_fps': 50,
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
agent = DDPGAgent(state_size=env.observation_space.shape[0], action_size=env.action_space.shape[0], random_seed=0)
video_dir = "videos_2"
video_freq = 10

def save_gravity_and_wind_to_csv(filename, episode, gravity, wind_power, average_score):
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        if episode == 1:
            writer.writerow(['Episode', 'Gravity', 'Wind Power', 'Average Score'])
        writer.writerow([episode, gravity, wind_power, average_score])

def train_ddpg(n_episodes=1000, max_t=1000, print_every=10):
    scores_deque = deque(maxlen=print_every)
    scores = []
    max_score = -np.inf
    best_episode = 0
    for i_episode in range(1, n_episodes + 1):
        rand_gravity = random.randint(-11, -5)
        rand_wind = random.randint(0, 20)
        env = gym.make("LunarLander-v2", continuous=True, gravity=rand_gravity, enable_wind=True, wind_power=rand_wind,
                       render_mode='rgb_array')

        # record video ever x episode
        if i_episode % 10 == 0:
            env = RecordVideo(env, video_dir, name_prefix=i_episode)

        state = env.reset()
        agent.reset()
        score = 0
        state = state[0]

        for t in range(max_t):
            action = agent.act(state)
            next_state, reward, done, _, __ = env.step(action.squeeze())
            agent.step(state, action, reward, next_state, done, episode=i_episode , decrease_sigma_every_n_episodes=10, delta_sigma=0.01)
            state = next_state
            score += reward
            if done:
                break

        env.close()

        scores_deque.append(score)
        scores.append(score)

        if score > max_score:
            max_score = score
            best_episode = i_episode
            torch.save(agent.actor_local.state_dict(), 'best_checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'best_checkpoint_critic.pth')
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)), end="")

        if i_episode % print_every == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
        save_gravity_and_wind_to_csv('gravity_wind_data.csv', i_episode, env.gravity, env.wind_power, np.mean(scores_deque))

    return scores

train_ddpg()






