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

env = gym.make('BreakoutDeterministic-v4')
num_of_actions = env.action_space.n

render = False


class BreakoutNeuralNet(tf.keras.Model):
    def __init__(self, outs):
        super(BreakoutNeuralNet, self).__init__()
        self.conv_1 = tf.keras.layers.Conv2D(32, 8, strides=4, activation="relu")
        self.conv_2 = tf.keras.layers.Conv2D(64, 4, strides=2, activation="relu")
        self.conv_3 = tf.keras.layers.Conv2D(64, 3, strides=1, activation="relu")
        self.flat = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(128, activation="relu")
        self.dense2 = tf.keras.layers.Dense(64, activation="relu")
        self.dense3 = tf.keras.layers.Dense(outs, dtype=tf.float32)  # No activation

    def call(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.flat(x)
        x = self.dense1(x)
        x = self.dense2(x)
        return self.dense3(x)


class ReplayBuffer:
    def __init__(self, maxlen):
        self.buffer = []
        self.maxlen = maxlen
        self.is_over = False

    def add(self, q_state, q_action, q_reward, q_next_state):
        if self.is_over:
            self.buffer[np.random.randint(0, self.maxlen - 1)] = (q_state, q_action, q_reward, q_next_state)
        else:
            self.buffer.append((q_state, q_action, q_reward, q_next_state))
            if len(self.buffer) >= self.maxlen:
                self.is_over = True

    def get_all(self):
        states = [x[0] for x in self.buffer if x is not None]
        actions = [x[1] for x in self.buffer if x is not None]
        rewards = [x[2] for x in self.buffer if x is not None]
        nexts = [x[3] for x in self.buffer if x is not None]

        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), np.array(nexts)

    def clear(self):
        self.buffer.clear()
        self.is_over = False


main_model = BreakoutNeuralNet(num_of_actions)
decision_model = BreakoutNeuralNet(num_of_actions)
#decision_model.compile(optimizer='adam', loss='mse')
decision_model.set_weights(main_model.get_weights())
replay_buffer = ReplayBuffer(150)
mse = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(1e-4)

# Hyper parameters
alpha = 0.1
gamma = 0.99
epsilon = 1

# For plotting metrics
episode_reward_history = []
load = 'breakout2020-12-31_06_17_05_261896.pickle'
load = None


def actor_action(a_state):
    scores = main_model(a_state)
    if random.uniform(0, 1) > epsilon:
        choice = np.argmax(scores)
    else:
        # wscores = softmax(scores[0])
        choice = random.choices(range(num_of_actions), weights=scores[0])[0]
    return choice


# todo use this as callback in model
def back_propagate(states, actions, rewards, next_states):
    # Get current Q_S (moved under tape)
    masks = tf.one_hot(actions, num_of_actions)

    # Predict the maximum reward from the next state
    # Get Q_PRIME_S
    next_scores = decision_model(next_states)
    print(next_scores.shape)
    next_scores = tf.reshape(next_scores, [150, num_of_actions])
    q_s_prime = tf.reduce_max(next_scores, axis=-1)

    with tf.GradientTape() as tape:
        q_s = main_model(states)
        q_s = tf.reshape(q_s, [150, num_of_actions])
        masks = tf.reduce_sum(masks * q_s, axis=1)

        # Back propagate the computed new value for the current state
        new_values = (1 - alpha) * masks + alpha * (rewards + gamma * q_s_prime)  # MASKS OR Q_S

        loss = mse(new_values, masks)

    # Apply changes on weigths
    grads = tape.gradient(loss, main_model.trainable_variables)
    optimizer.apply_gradients(zip(grads, main_model.trainable_variables))


#save_name = 'breakout' + str(datetime.now()).replace(' ', '_').replace('.', '_').replace(':', '_') + '.pickle'
save_name = "./drive/MyDrive/Colab Notebooks/" + "convrn"
for episode in range(0, 10000000):
    state = env.reset()
    env.step(1)

    reward, episode_reward = 0, 0
    previous_lives = 5
    done = False

    if load and episode == 0:
        save_name = load
        model = keras.models.load_model(save_name)
        print([x.shape for x in model.get_weights()])
        main_model(np.asarray([state]))
        main_model.set_weights(model.get_weights())
        decision_model(np.asarray([state]))
        decision_model.set_weights(main_model.get_weights())

    while not done:
        # Make a decision
        state = np.mean(state, 2, keepdims=False)
        state = state[35:195]
        state = state[::2, ::2]
        state = np.array([state, state, state, state])
        state = state.reshape((80, 80, 4))
        action = actor_action(np.asarray([state]))

        # Execute the action and get the new state
        next_state, reward, done, info = env.step(action)

        reward *= 3

        if action == 1 and info["ale.lives"] != previous_lives:
            reward += 0.5
            previous_lives = info["ale.lives"]

        episode_reward += reward

        # Store actions in replay buffer
        aux_state = next_state
        next_state = np.mean(next_state, 2, keepdims=False)
        next_state = next_state[35:195]
        next_state = next_state[::2, ::2]
        next_state = np.array([next_state, next_state, next_state, next_state])
        next_state = next_state.reshape((80, 80, 4))
        replay_buffer.add(state, action, reward, next_state)

        if render:
            sleep(0.01)
            env.render()

        state = aux_state

    if epsilon > 0.1:
        epsilon -= 0.000001

    episode_reward_history.append(episode_reward)
    running_reward = np.mean(episode_reward_history)

    if episode % 100 == 0:
        print(f"Episode: {episode}, mean: {running_reward}")
        states, actions, rewards, next_states = replay_buffer.get_all()
        back_propagate(states, actions, rewards, next_states)
        decision_model.set_weights(main_model.get_weights())
        replay_buffer.clear()

        main_model.save(save_name)
