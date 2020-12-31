from datetime import datetime

import gym
from time import sleep
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow import keras

env = gym.make('Breakout-ram-v0')
num_of_actions = env.action_space.n

render = True


class BreakoutNeuralNet(tf.keras.Model):
    def __init__(self, outs):
        super(BreakoutNeuralNet, self).__init__()
        self.dense1 = tf.keras.layers.Dense(128, activation="relu")
        self.dense2 = tf.keras.layers.Dense(64, activation="relu")
        self.dense3 = tf.keras.layers.Dense(outs, dtype=tf.float32)  # No activation

    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        return self.dense3(x)


main_model = BreakoutNeuralNet(num_of_actions)
decision_model = BreakoutNeuralNet(num_of_actions)
decision_model.compile(optimizer='adam', loss='mse')
decision_model.set_weights(main_model.get_weights())
mse = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(1e-4)

# Hyper parameters
alpha = 0.1
gamma = 0.99
epsilon = 1

# For plotting metrics
episode_reward_history = []
load = 'backu_breakout2020-12-31_06_38_34_631078.pickle'
# load = None


def actor_action(a_state):
    scores = main_model(a_state)
    choice = np.argmax(scores)
    return choice


state = env.reset()

done = False

model = keras.models.load_model(load)
print([x.shape for x in model.get_weights()])
main_model(np.asarray([state]))
main_model.set_weights(model.get_weights())
decision_model(np.asarray([state]))
decision_model.set_weights(main_model.get_weights())

for i in range(1000):
    state = env.reset()
    done = False

    counter = 0

    while not done:
        counter = counter + 1
        # Make a decision
        state = np.asarray([state])
        action = actor_action(state)

        # Execute the action and get the new state
        next_state, reward, done, info = env.step(action)
        
        if counter > 600:
            break

        if render:
            sleep(0.01)
            env.render()

        state = next_state
