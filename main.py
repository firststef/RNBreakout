import gym
from time import sleep
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input

env = gym.make('Breakout-ram-v0')
num_of_actions = env.action_space.n

render = False


class BreakoutNeuralNet(tf.keras.Model):
    def __init__(self, outs):
        super(BreakoutNeuralNet, self).__init__()
        self.dense1 = tf.keras.layers.Dense(128, activation="relu")
        self.dense2 = tf.keras.layers.Dense(64, activation="relu")
        self.dense3 = tf.keras.layers.Dense(outs, dtype=tf.float32)  # No activation

    def call(self, x):
        # x = tf.keras.backend.flatten(x)
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
decision_model.set_weights(main_model.get_weights())
replay_buffer = ReplayBuffer(150)
mse = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(1e-4)

# Hyper parameters
alpha = 0.1
gamma = 0.6
epsilon = 1

# For plotting metrics
all_epochs = []
all_penalties = []


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

        loss = mse(new_values, masks)

    # Apply changes on weigths
    grads = tape.gradient(loss, main_model.trainable_variables)
    optimizer.apply_gradients(zip(grads, main_model.trainable_variables))


for episode in range(0, 1000):
    state = env.reset()

    epochs, reward, = 0, 0
    done = False

    while not done:
        # Make a decision
        state = np.asarray([state])
        action = actor_action(state)

        # Execute the action and get the new state
        next_state, reward, done, info = env.step(action)

        # Store actions in replay buffer
        replay_buffer.add(state, action, reward, np.asarray([next_state]))

        if render:
            sleep(0.01)
            env.render()

        state = next_state
        epochs += 1

    if epsilon > 0.05:
        epsilon -= 0.001

    if episode % 1 == 0:
        print(f"Episode: {episode}")
        states, actions, rewards, next_states = replay_buffer.get_all()
        back_propagate(states, actions, rewards, next_states)
        decision_model.set_weights(main_model.get_weights())
        replay_buffer.clear()
