import random

from collections import deque

import numpy as np
import tensorflow as tf
import keras.backend as K

from keras.models import Sequential
from keras.models import load_model, clone_model
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import PReLU
from keras.optimizers import Adam
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation
from keras.losses import MSE
from keras import regularizers



def huber_loss(y_true, y_pred, clip_delta=5.0):
    """Huber loss - Custom Loss Function for Q Learning
    Links: 	https://en.wikipedia.org/wiki/Huber_loss
            https://jaromiru.com/2017/05/27/on-using-huber-loss-in-deep-q-learning/
    """
    error = y_true - y_pred
    cond = K.abs(error) <= clip_delta
    squared_loss = 0.5 * K.square(error)
    quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)
    return K.mean(tf.where(cond, squared_loss, quadratic_loss))


class Agent:
    """ Stock Trading Bot """

    def __init__(self, state_size, strategy="t-dqn", reset_every=1000, pretrained=False, model_name=None):
        self.strategy = strategy 

        # agent config
        self.state_size = 69 # normalized previous days used to be state_size ##zy change
        self.action_size = 2  # [sit, buy, sell]
        self.model_name = model_name
        self.inventory = []
        self.memory = deque(maxlen=10000)
        self.first_iter = True

        # model config
        self.model_name = model_name
        self.gamma = 0.8  # affinity for long term reward
        self.epsilon = 0.99
        self.epsilon_min = 0.02
        self.epsilon_decay = 0.998
        self.learning_rate = 0.001
        self.loss = huber_loss
        self.custom_objects = {"huber_loss": huber_loss}  # important for loading the model from memory
        self.learning_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
          self.learning_rate, decay_steps = 1000, decay_rate = 0.96, staircase=True, name=None
        )
        self.optimizer = Adam(lr=self.learning_rate) # beta_1 decay

        if pretrained and self.model_name is not None:
            self.model = self.load()
        else:
            self.model = self._model()

        # strategy config
        if self.strategy in ["t-dqn", "double-dqn"]:
            self.n_iter = 1
            self.reset_every = reset_every

            # target network
            self.target_model = clone_model(self.model)
            self.target_model.set_weights(self.model.get_weights())

    def _model(self):
        """Creates the model
        """
        model = Sequential()
        # model.add(Dropout(0.3))
        # model.add(Dense(units=256, activation="relu"))
        # model.add(Dense(units=128, activation="relu"))
        # model.add(Dropout(0.1))
        model.add(Dense(units=32, activation = 'relu'))
        # model.add(Dense(units=128))
        # model.add(PReLU())
        # model.add(Dropout(0.2))
        # model.add(Dense(units=64))
        # model.add(PReLU())
        model.add(Dense(units=16, activation = 'relu'))
        # model.add(Dropout(0.3))
        # model.add(Dropout(0.2))
        # model.add(Dense(units=32))
        # model.add(PReLU())
        model.add(Dense(units=8, activation = 'relu'))
        # model.add(Dropout(0.35))
        model.add(Dense(units=4, activation = 'relu'))
        # model.add(Dropout(0.35))
        # model.add(Dense(units=8, activation = 'relu'))
        # model.add(Dropout(0.4))
        # model.add(Dense(units=self.action_size))

        # model.add(Dropout(0.2))
        # model.add(Dense(units=256, activation="relu"))
        # model.add(Dropout(0.25))
        # model.add(Dense(units=64, activation="relu", kernel_regularizer=regularizers.l1_l2(l1=1e-3, l2=1e-3)))
        # model.add(Dense(units=5))
        # model.add(PReLU())
        # model.add(Dropout(0.2))
        # model.add(Dense(units=8, activation="relu"))

        # model.add(Dense(units=8, activation = 'relu'))
        # model.add(Dropout(0.5))
        # model.add(Dense(units=self.action_size))

        # model.add(Dropout(0.2))
        # model.add(Dense(units=256, activation="relu"))
        # model.add(Dropout(0.25))
        # model.add(Dense(units=64, activation="relu", kernel_regularizer=regularizers.l1_l2(l1=1e-3, l2=1e-3)))
        # model.add(Dense(units=5))
        # model.add(PReLU())
        # model.add(Dropout(0.2))
        # model.add(Dense(units=8, activation="relu"))
        model.add(Dense(units=self.action_size))

        

        model.compile(loss=self.loss, optimizer=self.optimizer)
        return model

    def remember(self, state, action, reward, next_state, done):
        """Adds relevant data to memory
        """
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, is_eval=False):
        """Take action from given possible set of actions
        """
        # 0.5.4 now we only have two actions buy and hold(all cash)
        # take random action in order to diversify experience at the beginning
        if not is_eval and random.random() <= self.epsilon:
            return random.randrange(self.action_size )   # only choose buy and sell

        if self.first_iter and not is_eval:
            self.first_iter = False
            return 1  # make a definite buy on the first iter

        action_probs = self.model.predict(state)
        # if not is_eval and np.argmax(action_probs[0]) != 0:
        #   print("make an action!")
        # return np.argmax(action_probs[0])
        if np.argmax(action_probs[0]) == 0:
          return np.argmax(action_probs[0]) 
        else:
          return np.argmax(action_probs[0])

    def train_experience_replay(self, batch_size):
        """Train on previous experiences in memory
        """
        mini_batch = random.sample(self.memory, batch_size)
        X_train, y_train = [], []

        # DQN
        if self.strategy == "dqn":
            for state, action, reward, next_state, done in mini_batch:
                if done:
                    target = reward
                else:
                    # approximate deep q-learning equation
                    target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])

                # estimate q-values based on current state
                q_values = self.model.predict(state)
                # update the target for current action based on discounted reward
                q_values[0][action] = target

                X_train.append(state[0])
                y_train.append(q_values[0])

        # DQN with fixed targets
        elif self.strategy == "t-dqn":
            # print(self.n_iter)
            if self.n_iter % self.reset_every == 0:
                # reset target model weights
                print('updated!!!!')
                self.target_model.set_weights(self.model.get_weights())
            self.n_iter += 1
            for state, action, reward, next_state, done in mini_batch:
                if done:
                    target = reward
                else:
                    # approximate deep q-learning equation with fixed targets
                    target = reward + self.gamma * np.amax(self.target_model.predict(next_state)[0])

                # estimate q-values based on current state
                q_values = self.model.predict(state)
                # update the target for current action based on discounted reward
                q_values[0][action] = target

                X_train.append(state[0])
                y_train.append(q_values[0])

        # Double DQN
        elif self.strategy == "double-dqn":
            if self.n_iter % self.reset_every == 0:
                # reset target model weights
                self.target_model.set_weights(self.model.get_weights())

            for state, action, reward, next_state, done in mini_batch:
                if done:
                    target = reward
                else:
                    # approximate double deep q-learning equation
                    target = reward + self.gamma * self.target_model.predict(next_state)[0][
                        np.argmax(self.model.predict(next_state)[0])]

                # estimate q-values based on current state
                q_values = self.model.predict(state)
                # update the target for current action based on discounted reward
                q_values[0][action] = target

                X_train.append(state[0])
                y_train.append(q_values[0])

        else:
            raise NotImplementedError()

        # update q-function parameters based on huber loss gradient
        loss = self.model.fit(
            np.array(X_train), np.array(y_train),
            epochs=1, verbose=0
        ).history["loss"][0]

        # as the training goes on we want the agent to
        # make less random and more optimal decisions
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return loss

    def save(self, episode):
        self.model.save("models/{}_{}".format(self.model_name, episode))

    def load(self):
        return load_model("models/" + self.model_name, custom_objects=self.custom_objects)
