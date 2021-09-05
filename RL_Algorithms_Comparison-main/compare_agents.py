import gym
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from bins_q_learning_agent import BinsQLearningAgent
from linear_q_learning_agent import LinearQLearningAgent
from linear_q_learning_rbf_agent import LinearQLearningRBFAgent
from deep_sarsa_agent import DSARSANAgent
from deep_q_learning_agent import DQNAgent
from linear_n_step_rbf_agent import LinearNStepRBFAgent
from td_lambda_rbf_agent import TDLambdaRBFAgent


def get_arguments():
    p = ArgumentParser(description='Comparing performance of diff RL agnets.')
    p.add_argument("-env", dest="env", action="store", default='MountainCar-v0',
                   help="Chosen environment to test the agents (for Ex. 'CartPol-v0').")
    p.add_argument("-algo", dest="algorithm", action="store", default='LinearQLearningAgent',
                   help="Chosen agent (for Ex. 'LinearQLearningAgent').")
    p.add_argument("-epi", dest="n_episodes", action="store", default=0,
                   help="Number of episodes to runthe environment (for Ex. 1000).")

    opts = p.parse_args()
    return opts

def calc_moving_avg(total_rewards, k_elements=100):
    N = len(total_rewards)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = total_rewards[max(0, t-k_elements):(t+1)].mean()
        
    return running_avg

def play_one_episode(env, agent, train=True, render=False):
    s = env.reset()
    done = False

    # it = 0 #
    total_reward = 0
    while not done:
        if render:
            env.render()

        a = agent.get_action(s)
        s_next, r, done, _ = env.step(a)
        # if done and it < 199: #
        #    r = -100 #

        if train:
            agent.train(s, a, r, s_next, done)

        # it += 1 #
        s = s_next
        total_reward += r
    return total_reward

def test_agent(env, agent):
    print('\nTest Agent:')
    agent.zero_epsilon()
    total_reward = play_one_episode(env, agent, train=False, render=True)
    print(f'total reward: {total_reward}')
    env.close()

def main(env, agent, n_episodes, test=False, verbose=False):

    total_rewards = np.empty(n_episodes)
    for t in range(n_episodes):
        total_reward = play_one_episode(env, agent)
        total_rewards[t] = total_reward
        agent.update_epsilon()

        if t % (n_episodes/10) == 0 and verbose:
            average_rewards = round(np.mean(total_rewards[max(0, int(t-(n_episodes/10))):t+1]),1)
            print(f'episode: {t}/{n_episodes}, average reward: {average_rewards}, epsilon: {agent.epsilon}')

    moving_avg = calc_moving_avg(total_rewards, n_episodes//10)

    if test:
        test_agent(env, agent)

    return moving_avg

def plot_results(agents, moving_avgs, load_moving_avgs=False):
    if load_moving_avgs:
        path = 'moving_avg\\'
        for agent in agents:
            moving_avg = np.load(path + agent.name + '.npy')
            moving_avgs.append(moving_avg)

    for i in range(len(agents)):
        plt.plot(moving_avgs[i], label=agents[i].name)
    plt.title('Comparing Algorithms')
    plt.ylabel('Rewards')
    plt.xlabel('Episodes')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    opts = get_arguments()
    env = gym.make(opts.env)

    agents = [ BinsQLearningAgent(env.action_space, env.observation_space),
               LinearQLearningAgent(env.action_space, env.observation_space),
               LinearQLearningRBFAgent(env.action_space, env.observation_space),
               DSARSANAgent(env.action_space, env.observation_space),
               DQNAgent(env.action_space, env.observation_space),
               LinearNStepRBFAgent(env.action_space, env.observation_space),
               TDLambdaRBFAgent(env.action_space, env.observation_space)
             ]

    moving_avgs = []
    for agent in agents:
        moving_avg = main(env, agent, n_episodes=opts.n_episodes, test=True, verbose=True)
        np.save(f'moving_avg/{agent.name}', moving_avg)
        moving_avgs.append(moving_avg)

    plot_results(agents, moving_avgs)
