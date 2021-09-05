import pickle
import numpy as np
from sklearn.kernel_approximation import RBFSampler
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion
from collections import deque

class FeatureTransformer:
    def __init__(self, observation_space, n_samples=10000):
        samples = [observation_space.sample() for _ in range(n_samples)]
        self.scaler = StandardScaler()
        self.scaler.fit(samples)

        self.featurizer = FeatureUnion([
                        ('rbf1', RBFSampler(gamma=5, n_components=500)),
                        ('rbf2', RBFSampler(gamma=2, n_components=500)),
                        ('rbf3', RBFSampler(gamma=1, n_components=500)),
                        ('rbf4', RBFSampler(gamma=0.5, n_components=500))
                        ])
        self.featurizer.fit(self.scaler.transform(samples))

    def transform(self, X):
        return self.featurizer.transform(self.scaler.transform(X))

class SGDRegressor:
    def __init__(self, n_features, learning_rate=0.001):
        self.learning_rate = learning_rate
        self.W = np.random.randn(n_features)

    def predict(self, X):
        return X.dot(self.W)
    
    def partial_fit(self, X, Y):
        self.W = self.W + self.learning_rate*X.T.dot((Y - self.predict(X)))

class LinearNStepRBFAgent:
    def __init__(self, action_space, observation_space, gamma=0.99,n_step=5, model_file_path=None):
        self.name = 'Linear N-Step RBF Agent'
        print(self.name + ' Created.')

        self.action_space = action_space
        self.observation_space = observation_space

        self.n_step = n_step
        self.gamma = gamma
        # array of [gamma^0, gamma^1, ..., gamma^(n-1)]
        self.multiplier = np.array([self.gamma]*self.n_step)**np.arange(self.n_step)

        self.states = deque()
        self.states_next = deque()
        self.actions = deque()
        self.rewards = deque()
        self.dones = deque()

        self.featurizer = FeatureTransformer(self.observation_space)

        self.epsilon = 1
        self.epsilon_decay = 0.05
        self.epsilon_min = 0.1

        if model_file_path is None:
            self.models = self.create_model()
        else:
            self.models = self.load_model(model_file_path)

    def create_model(self):
        models = []
        sample = self.observation_space.sample().reshape(1,-1)
        sample_featured = self.featurizer.transform(sample)
        for _ in range(self.action_space.n):
            model = SGDRegressor(sample_featured.shape[1])
            models.append(model)
        return models

    def get_action(self, s):
        p = np.random.random()
        if p < self.epsilon:
            return self.action_space.sample()
        else:
            return np.argmax(self.predict(s))

    def predict(self, s):
        s = s.reshape(1,-1)
        s_featured = self.featurizer.transform(s)
        return [self.models[i].predict(s_featured) for i in range(self.action_space.n)]

    def train(self, s, a, r, s_next, done):
        s = s.reshape(1,-1)
        s_featured = self.featurizer.transform(s)

        self.states.append(s_featured)
        self.actions.append(a)
        self.rewards.append(r)
        self.states_next.append(s_next)
        self.dones.append(done)

        if len(self.rewards) >= self.n_step:
            return_up_to_prediction = self.multiplier.dot(self.rewards)
            G = return_up_to_prediction + (self.gamma**self.n_step)*np.max(self.predict(self.states_next[0]))
            self.models[self.actions[0]].partial_fit(self.states[0], G)

            self.states.popleft()
            self.actions.popleft()
            self.rewards.popleft()
            self.states_next.popleft()
            self.dones.popleft()

        if self.states[-1][0][0] >= 0.5 or self.dones[-1]:
            while len(self.rewards) > 0:
                return_up_to_prediction = self.multiplier[:len(self.rewards)].dot(self.rewards)
                G = return_up_to_prediction + (self.gamma**self.n_step)*np.max(self.predict(s_next))
                self.models[self.actions[0]].partial_fit(self.states[0], G)

                self.states.popleft()
                self.actions.popleft()
                self.rewards.popleft()
                self.states_next.popleft()
                self.dones.popleft()
    
    def update_epsilon(self):
        self.epsilon = max(self.epsilon - self.epsilon_decay, self.epsilon_min)

    def zero_epsilon(self):
        self.epsilon = 0

    def save_model(self, model_file_path):
        for i in range(self.actions_space.n):
            temp = model_file_path.split('.')
            file_path = temp[0] + str(i) + '.' + temp[1]
            with open(file_path,'wb') as pkl_file:
                pickle.dump(self.models[i], pkl_file)

    def load_model(self, model_file_path):
        models = []
        for i in range(self.actions_space.n):
            temp = model_file_path.split('.')
            file_path = temp[0] + str(i) + '.' + temp[1]
            with open(file_path,'rb') as pkl_file:
                model = pickle.load(pkl_file)
                models.append(model)
        return models