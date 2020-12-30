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


def BreakoutNeuralNet():
    # Network defined by the Deepmind paper
    # inp = Input(shape=(84, 84, 4,))
    inp = Input(shape=(128, ))

    # Convolutions on the frames on the screen
    # layer1 = Conv2D(32, 8, strides=4, activation="relu")(input)
    # layer2 = Conv2D(64, 4, strides=2, activation="relu")(layer1)
    # layer3 = Conv2D(64, 3, strides=1, activation="relu")(layer2)

    # layer4 = Flatten()(layer3)

    layer5 = Dense(512, activation="relu")(inp)
    output = Dense(num_of_actions, activation="linear")(layer5)

    return Model(inputs=inp, outputs=output)


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


main_model = BreakoutNeuralNet()
decision_model = BreakoutNeuralNet()
decision_model.set_weights(main_model.get_weights())
tape = tf.GradientTape()
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
        choice = random.choice(range(num_of_actions), weights=scores[0])[0]
    return choice


# todo use this as callback in model
@tf.function
def back_propagate(states, actions, rewards, next_states):
    q_s = main_model(states)
    masks = tf.one_hot(actions, num_of_actions)
    masks = q_s * masks

    # Predict the maximum reward from the next state
    next_scores = decision_model(next_states)
    q_s_prime = np.array([max(row) for row in next_scores])

    # Back propagate the computed new value for the current state
    new_values = (1 - alpha) * q_s + alpha * (rewards + gamma * q_s_prime)

    loss = mse(new_values, masks)

    # Apply changes on weigths
    grads = tape.gradient(loss, main_model.trainable_variables)
    optimizer.apply_gradients(zip(grads, main_model.trainable_variables))


for episode in range(1, 1000):
    state = env.reset()

    epochs, reward, = 0, 0
    done = False

    while not done:
        # Make a decision
        action = actor_action(state)

        # Execute the action and get the new state
        next_state, reward, done, info = env.step(action)

        # Store actions in replay buffer
        replay_buffer.add(state, action, reward, next_state)

        if render:
            sleep(0.01)
            env.render()

        state = next_state
        epochs += 1

    if epsilon > 0.05:
        epsilon -= 0.001

    if episode % 100 == 0:
        print(f"Episode: {episode}")
        states, actions, rewards, next_states = replay_buffer.get_all()
        back_propagate(states, actions, rewards, next_states)
        decision_model.set_weights(main_model.get_weights())
        replay_buffer.clear()
