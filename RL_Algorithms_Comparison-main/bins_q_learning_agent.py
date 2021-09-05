import pickle
import numpy as np


class FeatureTransformer:
    def __init__(self, observation_space, n_bins, n_samples=10000):
        self.dimension = observation_space.shape[0]

        max_values = observation_space.high
        min_values = observation_space.low

        self.bins_list = [np.linspace(-2.4, 2.4, n_bins-1),
                          np.linspace(-2, 2, n_bins-1),
                          np.linspace(-0.4, 0.4, n_bins-1),
                          np.linspace(-3.5, 3.5, n_bins-1)]

        #self.bins_list = [np.linspace(max_values[i], min_values[i], n_bins-1) for i in range(self.dimension)]

    def transform(self, X):
        key = ''
        for i in range(self.dimension):
            key += str(np.digitize(X[i], self.bins_list[i]))
        return key


class BinsQLearningAgent:
    def __init__(self, action_space, observation_space, learning_rate=0.001, gamma=0.99, n_bins=10, model_file_path=None):
        self.name = 'Bins Q Learning Agent'
        print(self.name + ' Created.')

        self.action_space = action_space
        self.observation_space = observation_space

        self.learning_rate = learning_rate
        self.gamma = gamma
        self.n_bins = n_bins

        self.epsilon = 1
        self.epsilon_decay = 0.05
        self.epsilon_min = 0.1

        self.featurizer = FeatureTransformer(self.observation_space, self.n_bins)

        if model_file_path is None:
            self.Q = self.create_model()
        else:
            self.Q = self.load_model(self)

    def create_model(self):
        Q = {}
        n_Q_values = self.n_bins**self.featurizer.dimension
        for i in range(n_Q_values):
            Q[str(i).zfill(len(str(n_Q_values))-1)] = np.zeros(self.action_space.n)
        return Q

    def get_action(self, s):
        p = np.random.random()
        if p < self.epsilon:
            return self.action_space.sample()
        else:
            return np.argmax(self.predict(s))

    def predict(self, s):
        s_featured = self.featurizer.transform(s)
        return self.Q[s_featured]

    def train(self, s, a, r, s_next, done):
        s = self.featurizer.transform(s)

        self.Q[s][a] = self.Q[s][a] + self.learning_rate * (r + self.gamma*np.max(self.predict(s_next)) - self.Q[s][a])

    def update_epsilon(self):
        self.epsilon = max(self.epsilon - self.epsilon_decay, self.epsilon_min)

    def zero_epsilon(self):
        self.epsilon = 0

    def save_model(self, model_file_path):
        pass

    def load_model(self, model_file_path):
        pass