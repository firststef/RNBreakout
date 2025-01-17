from datetime import datetime

import gym
from time import sleep
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras

env = gym.make('Breakout-ram-v0')
num_of_actions = env.action_space.n
num_of_history_actions = 3

render = False
save_name = "./drive/MyDrive/Colab Notebooks/" + 'breakout' + str(datetime.now()).replace(' ', '_').replace('.', '_').replace(':', '_')


class BreakoutNeuralNet(tf.keras.Model):
    def __init__(self, outs):
        super(BreakoutNeuralNet, self).__init__()
        self.dense1 = tf.keras.layers.Dense(128 * num_of_history_actions, activation="relu")
        self.dense2 = tf.keras.layers.Dense(64, activation="relu")
        self.dense3 = tf.keras.layers.Dense(outs, dtype=tf.float32)  # No activation

    def call(self, x):
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


class CyclingBuffer:
    def __init__(self, maxlen):
        self.buffer = []
        self.maxlen = maxlen
        self.is_over = False

    def add(self, elem):
        if self.is_over:
            self.buffer = self.buffer[1:]
            self.buffer.append(elem)
        else:
            self.buffer.append(elem)
            if len(self.buffer) >= self.maxlen:
                self.is_over = True


main_model = BreakoutNeuralNet(num_of_actions)
decision_model = BreakoutNeuralNet(num_of_actions)
decision_model.compile(optimizer='adam', loss='mse')
decision_model.set_weights(main_model.get_weights())
replay_buffer = ReplayBuffer(150)
mse = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(1e-4)
states_c = CyclingBuffer(num_of_history_actions)
next_states_c = CyclingBuffer(num_of_history_actions)
loss_function = keras.losses.Huber()

# Hyper parameters
alpha = 0.1
gamma = 0.99
epsilon = 0.6

# For plotting metrics
episode_reward_history = []
# load = "./drive/MyDrive/Colab Notebooks/" + 'backup_breakout2020-12-31_06_38_34_631078.pickle'
# load = None
save_name = "./drive/MyDrive/Colab Notebooks/" + 'back3_simple'
load = save_name


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
    next_scores = tf.reshape(next_scores, [150, 4])
    q_s_prime = tf.reduce_max(next_scores, axis=-1)

    with tf.GradientTape() as tape:
        q_s = main_model(states)
        q_s = tf.reshape(q_s, [150, 4])
        masks = tf.reduce_sum(masks * q_s, axis=1)

        # Back propagate the computed new value for the current state
        new_values = (1 - alpha) * masks + alpha * (rewards + gamma * q_s_prime)  # MASKS OR Q_S

        loss = loss_function(new_values, masks)

    # Apply changes on weigths
    grads = tape.gradient(loss, main_model.trainable_variables)
    optimizer.apply_gradients(zip(grads, main_model.trainable_variables))


for episode in range(0, 100000):
    state = env.reset()
    next_state = state

    reward, episode_reward, previous_lives = 0, 0, 0
    episode_reward_history.clear()
    done = False

    # Filling array at start
    for i in range(num_of_history_actions):
        states_c.add(state)
        next_states_c.add(state)

    if load and episode == 0:
        model = keras.models.load_model(load)
        print([x.shape for x in model.get_weights()])
        state = np.asarray([states_c.buffer])
        state = state.reshape((1, 128 * num_of_history_actions))
        main_model(state)
        main_model.set_weights(model.get_weights())
        decision_model(state)
        decision_model.set_weights(main_model.get_weights())
        state = env.reset()

    while not done:
        states_c.add(state)
        state = np.asarray([states_c.buffer])
        state = state.reshape((1, 128 * num_of_history_actions))
        action = actor_action(state)

        # Execute the action and get the new state
        next_state, reward, done, info = env.step(action)

        episode_reward += reward
        next_states_c.add(next_state)

        # Store actions in replay buffer
        save_new = np.asarray([next_states_c.buffer])
        save_new = save_new.reshape((1, 128 * num_of_history_actions))
        replay_buffer.add(state, action, reward, save_new)

        # if render:
        #     sleep(0.01)
        #     env.render()

        state = next_state

    if epsilon > 0.5:
        epsilon -= 0.00001

    episode_reward_history.append(episode_reward)
    running_reward = np.mean(episode_reward_history)

    if episode % 100 == 0:
        print(f"Episode: {episode}, mean: {running_reward}")
        states, actions, rewards, next_states = replay_buffer.get_all()
        back_propagate(states, actions, rewards, next_states)
        decision_model.set_weights(main_model.get_weights())
        replay_buffer.clear()

        # print(main_model.get_weights())
        main_model.save(save_name)
