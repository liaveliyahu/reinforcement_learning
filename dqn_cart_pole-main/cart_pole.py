import gym
import numpy as np
import matplotlib.pyplot as plt
from DQN_agent import DQNAgent


MODEL_PATH = 'dqn_model/my_model.h5'
N_EPISODES = 400


def calc_moving_avg(total_rewards, k_elements=100):
    total_rewards = np.array(total_rewards)
    N = len(total_rewards)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = total_rewards[max(0, t-k_elements):(t+1)].mean()
        
    return running_avg

def play_one_episode(env, agent, train=True, render=False):
    s = env.reset()
    done = False

    step = 0
    reward = 0
    while not done:
        if render:
            env.render()

        a = agent.get_action(s)
        s_next, r, done, _ = env.step(a)

        # adjust reward
        if done and step < 199:
            r = -100

        if train:
            agent.train(s, a, r, s_next, done)

        s = s_next
        step += 1
        reward += r
    
    return step, reward

def main(n_episodes):
    env = gym.make('CartPole-v0')
    agent = DQNAgent(env.action_space, env.observation_space,load_model_path=MODEL_PATH)

    steps = []
    rewards = []
    for it in range(n_episodes):
        step, reward = play_one_episode(env, agent)
        steps.append(step)
        rewards.append(reward)
        agent.update_epsilon()
        agent.save_model(MODEL_PATH)

        if it % (n_episodes/10) == 0:
            print(f'epsiode: {it}/{n_episodes}, steps: {step}, reward: {reward}, epsilon: {agent.epsilon}')
    
    moving_avgs = calc_moving_avg(rewards, n_episodes//10)

    plt.plot(steps, label='total steps')
    plt.plot(reward, label='total rewards')
    plt.plot(moving_avgs, label='moving average')
    plt.title('Performance')
    plt.xlabel('Episodes')
    plt.ylabel('Steps')
    plt.legend()
    plt.show()

    print('Test Agent:')
    agent.zero_epsilon()
    steps, reward = play_one_episode(env, agent, train=False, render=True)
    print(f'steps: {steps}, reward: {reward}')
    env.close()

if __name__ == '__main__':
    main(n_episodes=N_EPISODES)