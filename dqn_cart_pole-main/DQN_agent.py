import numpy as np
import tensorflow as tf


class DQNAgent:
    def __init__(self, action_space, observation_space, load_model_path=None):
        self.action_space = action_space
        self.observation_space = observation_space

        self.epsilon = 1
        self.epsilon_decay = 0.01
        self.epsilon_min = 0.1

        self.gamma = 0.99

        if load_model_path is None:
            self.model = self._create_model()
        else:
            self.model = self.load_model(load_model_path)

    def _create_model(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(32, activation='relu', input_shape=self.observation_space.shape))
        model.add(tf.keras.layers.Dense(32, activation='relu'))
        model.add(tf.keras.layers.Dense(self.action_space.n, activation='linear'))
        model.compile(loss='mse',optimizer=tf.keras.optimizers.Adam())
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
            target = r + self.gamma*np.max(self.predict(s_next))

        predicted_Q[0, a] = target
        self.model.fit(s, predicted_Q, batch_size=1, verbose=0)

    def update_epsilon(self):
        self.epsilon = max(self.epsilon-self.epsilon_decay,self.epsilon_min)

    def zero_epsilon(self):
        self.epsilon = 0

    def save_model(self, filepath):
        self.model.save(filepath)

    def load_model(self, filepath):
        return tf.keras.models.load_model(filepath)