# TD_Lambda_mountain_car
<p align="left">
  <img src="Assets\openaigym_td_lambda_mountain_car_agent.gif" width="450">
</p>

## Environment
A car is on a one-dimensional track, positioned between two "mountains". The goal is to drive up the mountain on the right; however, the car's engine is not strong enough to scale the mountain in a single pass. Therefore, the only way to succeed is to drive back and forth to build up momentum.

## Agent
TD(λ) is a generic reinforcement learning method that unifies both Monte Carlo simulation and 1-step TD method.
TD(λ) improves over the offline λ-return algorithm in three ways. First it updates
the weight vector on every step of an episode rather than only at the end, and thus
its estimates may be better sooner. Second, its computations are equally distributed
in time rather than all at the end of the episode. And third, it can be applied to
continuing problems rather than just to episodic problems. In this section we present the
semi-gradient version of TD(λ) with function approximation.

## Results
<p align="left">
  <img src="Assets\Results.png" width="450">
</p>
TD(λ) combined with RBF kernel is very efficient algorithm for this environment.
It managed to solve the control problem after only 55 episodes. 
In addition it succeeds to improve the performance and reach the goal in fewer and fewer steps.

<p align="left">
  <img src="Assets\openaigym_td_lambda_mountain_car_agent_trained.gif" width="450">
</p>
In the above video we can see the agent reached to the goal after 4 swings.
With a bit more training it can do it with 3, but I provided only 400 episodes for this environment.

## References
* Gym OpenAI: https://gym.openai.com/envs/MountainCar-v0/
* Reinforcement Learning — TD(λ) Introduction(1): https://towardsdatascience.com/reinforcement-learning-td-%CE%BB-introduction-686a5e4f4e60
* Reinforcement Learning An introduction (Second Edition) by Richard S. Sutton and Andrew G. Barto
