# dqn_cart_pole
<p align="center">
  <img src="Assets\openaigym_cartpole_dqn_agent.gif" width="450">
</p>

## Environment
A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The system is controlled by applying a force of +1 or -1 to the cart. The pendulum starts upright, and the goal is to prevent it from falling over. A reward of +1 is provided for every timestep that the pole remains upright. The episode ends when the pole is more than 15 degrees from vertical, or the cart moves more than 2.4 units from the center.

## Agent
The DQN (Deep Q-Network) algorithm was developed by DeepMind in 2015. It was able to solve a wide range of Atari games (some to superhuman level) by combining reinforcement learning and deep neural networks at scale. The algorithm was developed by enhancing a classic RL algorithm called Q-Learning with deep neural networks and a technique called experience replay.

## Results
<p align="left">
  <img src="Assets\Results.jpeg" width="450">
</p>
After around 80 episodes the agent managed to reach the max steps (200) of the enviroment for the first time.
Immediately we can see a decrease in performance, and then the agent improves the performance and gets to the point where in most episodes it wins the game - reached 200 steps without dropping the pole.

## References
* Gym OpenAI: https://gym.openai.com/envs/CartPole-v0/
* TensorFlow: https://www.tensorflow.org/agents/tutorials/0_intro_rl


