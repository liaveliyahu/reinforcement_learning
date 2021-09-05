import numpy as np
import pickle

class SGDRegressor:
    def __init__(self, n_features, learning_rate=0.001):
        self.learning_rate = learning_rate
        self.W = np.random.randn(n_features) / np.sqrt(n_features)

    def predict(self, X):
        return X.dot(self.W)

    def partial_fit(self, X ,Y):
        self.W = self.W + self.learning_rate*(Y - self.predict(X)).dot(X)

class LinearQLearningAgent:
    def __init__(self, action_space, observation_space, gamma=0.99, model_file_path=None):
        self.name = 'Linear Q Learinig Agent'
        print(self.name + ' Created.')

        self.action_space = action_space
        self.observation_space = observation_space

        self.gamma = gamma

        self.epsilon = 1
        self.epsilon_decay = 0.05
        self.epsilon_min = 0.1

        if model_file_path is None:
            self.models = self.create_model()
        else:
            self.models = self.load_model(model_file_path)

    def create_model(self):
        models = []
        for _ in range(self.action_space.n):
            model = SGDRegressor(self.observation_space.shape[0])
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
        return [self.models[i].predict(s) for i in range(self.action_space.n)]

    def train(self, s, a, r, s_next, done):
        s = s.reshape(1,-1)
        if done:
            target = r
        else:
            target = r + self.gamma * np.max(self.predict(s_next))

        self.models[a].partial_fit(s, target)

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