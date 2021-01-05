import gym
from time import sleep
import random
import numpy as np
import tensorflow as tf
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
        self.dense1 = tf.keras.layers.Dense(512, activation="relu")
        self.dense2 = tf.keras.layers.Dense(outs, dtype=tf.float32)

    def call(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.flat(x)
        x = self.dense1(x)
        return self.dense2(x)


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


# Hyper parameters
alpha = 0.1
gamma = 0.99
epsilon = 1
num_of_history_actions = 4
update_models = 10000

main_model = BreakoutNeuralNet(num_of_actions)
decision_model = BreakoutNeuralNet(num_of_actions)
decision_model.set_weights(main_model.get_weights())
replay_buffer = ReplayBuffer(150)
huber = keras.losses.Huber()
optimizer = tf.keras.optimizers.Adam(1e-4)
states_c = CyclingBuffer(num_of_history_actions)
next_states_c = CyclingBuffer(num_of_history_actions)


# For plotting metrics
episode_reward_history = []
load = 'breakout2020-12-31_06_17_05_261896.pickle'
save_name = 'breakout'
load = None


def actor_action(a_state):
    scores = main_model(a_state)
    if random.uniform(0, 1) > epsilon:
        choice = np.argmax(scores)
    else:
        choice = random.choices(range(num_of_actions))[0]
    return choice


def back_propagate(states, actions, rewards, next_states):
    # Get current Q_S (moved under tape)
    masks = tf.one_hot(actions, num_of_actions)

    # Predict the maximum reward from the next state
    # Get Q_PRIME_S
    next_scores = decision_model(next_states)
    next_scores = tf.reshape(next_scores, [150, num_of_actions])
    q_s_prime = tf.reduce_max(next_scores, axis=-1)

    with tf.GradientTape() as tape:
        q_s = main_model(states)
        q_s = tf.reshape(q_s, [150, num_of_actions])
        q_s = tf.reduce_sum(masks * q_s, axis=1)

        # Back propagate the computed new value for the current state
        new_values = (1 - alpha) * q_s + alpha * (rewards + gamma * q_s_prime)

        loss = huber(new_values, q_s)

    # Apply changes on weigths
    grads = tape.gradient(loss, main_model.trainable_variables)
    optimizer.apply_gradients(zip(grads, main_model.trainable_variables))


for episode in range(0, 10000000):
    state = env.reset()
    env.step(1)

    reward, episode_reward = 0, 0
    previous_lives = 5
    done = False

    # Filling array at start
    b_state = state
    b_state = np.mean(b_state, 2, keepdims=False)
    b_state = b_state[35:195]
    b_state = b_state[::2, ::2]
    for i in range(num_of_history_actions):
        states_c.add(b_state)
        next_states_c.add(b_state)

    if load and episode == 0:
        model = keras.models.load_model(save_name)
        print([x.shape for x in model.get_weights()])
        state = np.array(states_c.buffer)
        state = state.reshape((80, 80, 4))
        main_model(np.asarray([state]))
        main_model.set_weights(model.get_weights())
        decision_model(np.asarray([state]))
        decision_model.set_weights(main_model.get_weights())
        continue

    while not done:
        # Make a decision
        state = np.mean(state, 2, keepdims=False)  # RGB to grayscale
        state = state[35:195]
        state = state[::2, ::2]
        states_c.add(state)
        state = np.array(states_c.buffer)
        state = state.reshape((80, 80, 4))
        state = np.asarray([state])
        action = actor_action(state)

        # Execute the action and get the new state
        next_state, reward, done, info = env.step(action)

        # reward *= 3
        # if action == 1 and info["ale.lives"] != previous_lives:
        #     reward += 0.5
        #     previous_lives = info["ale.lives"]

        episode_reward += reward

        # Store actions in replay buffer
        aux_state = next_state
        next_state = np.mean(next_state, 2, keepdims=False)
        next_state = next_state[35:195]
        next_state = next_state[::2, ::2]
        next_states_c.add(next_state)
        next_state = np.array(next_states_c.buffer)
        next_state = next_state.reshape((80, 80, 4))
        next_state = np.asarray([next_state])
        replay_buffer.add(state, action, reward, next_state)

        if render:
            sleep(0.01)
            env.render()

        state = aux_state

    if epsilon > 0.1:
        epsilon -= 0.000001

    if len(episode_reward_history) >= 100:
        episode_reward_history = episode_reward_history[1:100]
    episode_reward_history.append(episode_reward)

    if len(replay_buffer.buffer) == 150 and episode % 2 == 0:
        running_reward = np.mean(episode_reward_history)
        print(f"Episode: {episode}, mean: {running_reward}")
        states, actions, rewards, next_states = replay_buffer.get_all()
        back_propagate(states, actions, rewards, next_states)
        replay_buffer.clear()

    if episode % update_models == 0:
        decision_model.set_weights(main_model.get_weights())
        main_model.save(save_name)
