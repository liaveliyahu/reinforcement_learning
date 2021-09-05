import numpy as np
from tensorflow.keras import Sequential, models
from tensorflow.keras.layers import Dense, InputLayer
from tensorflow.keras.optimizers import Adam


class DQNAgent:
    def __init__(self, action_space, observation_space, gamma=0.99, model_file_path=None):
        self.name = 'Deep Q Network Agent'
        print(self.name + ' Created.')

        self.action_space = action_space
        self.observation_space = observation_space

        self.gamma = gamma

        self.epsilon = 1
        self.epsilon_decay = 0.05
        self.epsilon_min = 0.1

        if model_file_path is None:
            self.model = self.create_model(self.observation_space.shape, self.action_space.n)
        else: 
            self.model = self.load_model(model_file_path)

    def create_model(self, n_inputs, n_outputs):
        model = Sequential()
        model.add(InputLayer(input_shape=n_inputs))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(n_outputs, activation='linear'))
        model.compile(optimizer=Adam(), loss='mse')
        return model

    def get_action(self, s):
        p = np.random.random()
        if p < self.epsilon:
            return self.action_space.sample()
        else:
            return np.argmax(self.predict(s))

    def predict(self, s):
        s = s.reshape(1,-1)
        return self.model.predict(s)

    def train(self, s, a, r, s_next, done):
        s = s.reshape(1,-1)
        predicted_Q = self.predict(s)
        if done:
            target = r
        else:
            target = r + self.gamma * np.max(self.predict(s_next))
        
        predicted_Q[0, a] = target
        self.model.fit(s, predicted_Q, batch_size=1, verbose=0)

    def update_epsilon(self):
        self.epsilon = max(self.epsilon - self.epsilon_decay, self.epsilon_min)

    def zero_epsilon(self):
        self.epsilon = 0

    def save_model(self, model_file_path):
        self.model.save(model_file_path)

    def load_model(self, model_file_path):
        return models.load_model(model_file_path) 