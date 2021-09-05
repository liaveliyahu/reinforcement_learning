import gym
import numpy as np
import matplotlib.pyplot as plt
from td_lambda_rbf_agent import TDLambdaRBFAgent


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

    total_rewards = 0
    while not done:
        if render:
            env.render()

        a = agent.get_action(s)
        s_next, r, done, _ = env.step(a)

        if train:
            agent.train(s, a, r, s_next, done)

        s = s_next
        total_rewards += r
    
    return total_rewards

def main(n_episodes):
    env = gym.make('MountainCar-v0')
    agent = TDLambdaRBFAgent(env.action_space, env.observation_space)

    rewards = []
    for it in range(n_episodes):
        total_rewards = play_one_episode(env, agent)
        rewards.append(total_rewards)
        agent.update_epsilon()

        if it % (n_episodes/10) == 0:
            print(f'Epsiode: {it}/{n_episodes}, Reward: {total_rewards}, Epsilon: {agent.epsilon}')
    
    moving_avgs = calc_moving_avg(rewards, n_episodes//10)

    plt.plot(rewards, label='total rewards')
    plt.plot(moving_avgs, label='moving average')
    plt.title('Performance')
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.legend()
    plt.show()

    print('Test Agent:')
    agent.zero_epsilon()
    # env = gym.wrappers.Monitor(env, "Assets/", force=True)
    steps, reward = play_one_episode(env, agent, train=False, render=True)
    print(f'steps: {steps}, reward: {reward}')
    env.close()

if __name__ == '__main__':
    main(n_episodes=N_EPISODES)