# Comparison between Reinforcement Learning algorithms
## About 
This repository compares between many Reinforcement Learning Algorithms written from scratch. The environment is Mountain Car by OpenAI GYM: https://gym.openai.com/envs/MountainCar-v0/.
For each algorithm 1000 epsiodes were tested and the results were compared as moving average rewards.
<p align="center">
  <img src="Assets\openaigym._mountaincar_video.gif" width="450">
</p>

## Models
### Q Learning with Bins
Classic Q learning algorithm that uses Bins in order to get finite values for tabular method.

### Linear Q Learning
Q learning algorithm that uses linear regression to approximate the action function.

### Linear Q Learning with RBF Kernel
Same algorithm as 'Linear Q Learning' but with RBF kerenl to featurize the inputs.

### Deep SARSA Network
SARSA algorithm that uses deep neural network to approximate the action function.

### Deep Q Network (DQN)
Q learning algorithm that uses deep neural network to approximate the action function.

### Linear N-Step with RBF Kernel
N-Step TD algorithm with RBF kerenl to featurize the inputs.

### TD(λ) with RBF Kernel
TD(λ) algorithm with RBF kerenl to featurize the inputs.

## Results
<p align="left">
  <img src="Assets\Results.png" width="450">
</p>
As the graph above we can see that the fastest and accurate algorithm to win this environmnet is TD(λ).
Some of the algorithms even couldn't solve this problem and their moving average rewards remained on 0 for the whole episodes.
<p align="left">
  <img src="Assets\openaigym_mountaincar_video_trained.gif" width="450">
</p>

## References
Based on the online courses of Lazy Programmer:
* Artificial Intelligence: Reinforcement Learning in Python - https://www.udemy.com/course/artificial-intelligence-reinforcement-learning-in-python
* Advanced AI: Deep Reinforcement Learning in Python - https://www.udemy.com/course/deep-reinforcement-learning-in-python/
