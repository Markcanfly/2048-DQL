# https://keras.io/examples/rl/deep_q_network_breakout/
# https://towardsdatascience.com/deep-q-learning-tutorial-mindqn-2a4c855abffc

import numpy as np
import tensorflow as tf
from tensorflow.python.types.core import Value
import gym_2048
import gym
import random
from evaluate import largest_tile, summary
from tqdm import trange
from typing import List, Tuple

import textwrap
from datetime import datetime, timedelta

# Set hyperparameters
N_EPISODES = 100
N_TEST = 300
ACTIVATION = 'relu'
OUTPUT_ACTIVATION = 'linear'
BELLMAN_DISCOUNT = 0.72
LEARNING_RATE = 0.0001
MIN_EPSILON = 0.001
MAX_EPSILON = 1
EPSILON_DECAY = 0.005
BATCH_SIZE = 64
TRAIN_STEP = 12
# Neuron numbers for the hidden layers
L1 = 1024
L2 = 512
L3 = 256

begin = datetime.now()
with tf.device("/cpu:0"):
    def new_model() -> tf.keras.models.Sequential:
        init = tf.keras.initializers.GlorotUniform()
        model = tf.keras.models.Sequential([
            tf.keras.layers.InputLayer(input_shape=(None, 16)),
            tf.keras.layers.Dense(L1, ACTIVATION, kernel_initializer=init),
            tf.keras.layers.Dense(L2, ACTIVATION, kernel_initializer=init),
            tf.keras.layers.Dense(L3, ACTIVATION, kernel_initializer=init),
            tf.keras.layers.Dense(4, OUTPUT_ACTIVATION)
        ])
        model.compile(loss=tf.keras.losses.Huber(), optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), metrics=['accuracy'])
        return model
    class Agent():
        def __init__(self, env):
            self.env = env
            self.model = new_model()
            self.replay_memory = []
            self.offset_index = 0
        def reset_memory(self):
            self.replay_memory = []
        def action(self, training=False) -> int:
            if len(self.replay_memory) > 0 and np.array_equal(self.replay_memory[-1][0], self.replay_memory[-1][3]):
                self.offset_index = (self.offset_index + 1) % 4 # Roll over
            else:
                self.offset_index = 0
            action_q_values = self.model(self.env.board.reshape([1, 16]), training=training).numpy()
            # (action, q) pairs
            actions_sorted_by_q: List[Tuple[int,float]] = sorted(list(enumerate(action_q_values[0])), reverse=True, key=lambda x: x[1])
            # Offset is number of times the same board has been seen
            action = actions_sorted_by_q[self.offset_index][0]
            # Make sure we don't repeat the previous action (this could happen if it was random)
            if self.offset_index > 0 and action == self.replay_memory[-1][1]:
                action = actions_sorted_by_q[0][0]
            return action
        def step(self, training=False, epsilon=None) -> bool:
            '''Takes a step, and returns whether done'''
            if training and epsilon is None:
                raise ValueError('Must specify epsilon when training')
            
            if training and random.random() < epsilon:
                action = random.randint(0, 3)
            else:
                action = self.action(training=training)
            state = self.env.board
            state_next, reward, done, _ = self.env.step(action)
            if training:
                self.replay_memory.append([state, action, reward, state_next, done])
            else:
                self.replay_memory = [[state, action, reward, state_next, done]] # Only used to check if previous move yielded change
            return done
    
    def random_state() -> np.array:
        f = lambda n : 2**n
        state = f(np.random.randint(low=0, high=7, size=(1,16), dtype='int64'))
        for i in range(len(state[0])):
            if state[0][i] == 1:
                state[0][i] = 0
        return state
    
    def train(replay_memory, model, target_model):
        learning_rate = LEARNING_RATE
        discount_factor = BELLMAN_DISCOUNT

        if len(replay_memory) < BATCH_SIZE * 2:
            return

        batch_size = BATCH_SIZE
        mini_batch = random.sample(replay_memory, batch_size)
        current_states = np.array([tf.convert_to_tensor(transition[0].reshape(1, 16)) for transition in mini_batch])
        current_qs_list = model.predict_on_batch(current_states)
        new_current_states = np.array([tf.convert_to_tensor(transition[3].reshape(1, 16)) for transition in mini_batch])
        future_qs_list = target_model.predict_on_batch(new_current_states)

        X = []
        Y = []
        # Calculate updated Q values using the Bellman equation
        for index, (state, action, reward, _, done) in enumerate(mini_batch):
            if not done:
                max_future_q = reward + discount_factor * np.max(future_qs_list[index])
            else:
                max_future_q = reward

            current_qs = current_qs_list[index][0]
            current_qs[action] = (1 - learning_rate) * current_qs[action] + learning_rate * max_future_q

            X.append(state.flatten())
            Y.append(current_qs)
        model.fit(np.array(X), np.array(Y), batch_size=batch_size, verbose=0, shuffle=True)

    env = gym.make('2048-v0')
    env.seed(42)

    env.reset()
    env.render()

    # Train this with each move
    agent = Agent(env)
    # Update every nth step
    target_model = new_model()
    
    epsilon = MAX_EPSILON # copy from hyperparam

    stats_train = {}
    manual_abort = False
    for episode in trange(N_EPISODES):
        agent.reset_memory()
        state = env.reset()
        episode_reward = 0
        if manual_abort:
            break
        done = False
        step_count = 0
        while not done:
            agent.step(training=True, epsilon=epsilon)
            
            epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * np.exp(-EPSILON_DECAY * episode)
            
            done = agent.replay_memory[-1][4]
            reward = agent.replay_memory[-1][2]
            episode_reward += reward

            if step_count % TRAIN_STEP == 0 or done:
                train(agent.replay_memory, agent.model, target_model)

            max_tile = largest_tile(agent.env.board)
            if done:
                # Add to stats
                stats_train[max_tile] = stats_train.get(max_tile, 0) + 1
                if step_count >= BATCH_SIZE:
                    # Copy over weights to target model
                    target_model.set_weights(agent.model.get_weights())
                break
            step_count += 1

    training_end = datetime.now()
    print(summary(stats_train))

    stats_test = {}
    # Test
    
    for episode in trange(N_TEST):
        env.reset()
        agent.reset_memory()
        done = False
        step_count = 0
        while not done:
            agent.step()
            done = agent.replay_memory[-1][4]
            step_count += 1
            if done:
                # Add to stats
                max_tile = largest_tile(agent.env.board)
                stats_test[max_tile] = stats_test.get(max_tile, 0) + 1
                break
    testing_end = datetime.now()

    # Log hyperparameters, results
    with open('logs.txt', mode='a', encoding='utf8') as logfile:
        logfile.write(textwrap.dedent(f'''
# Hyperparameters:
N_EPISODES: {N_EPISODES}
N_TEST: {N_TEST}
L1: {L1}
L2: {L2}
L3: {L3}
ACTIVATION: {ACTIVATION}
OUTPUT_ACTIVATION: {OUTPUT_ACTIVATION}
LEARNING_RATE: {LEARNING_RATE}
MIN_EPSILON: {MIN_EPSILON}
MAX_EPSILON: {MAX_EPSILON}
EPSILON_DECAY: {EPSILON_DECAY}
BATCH_SIZE: {BATCH_SIZE}
# Results: 
Training time: {training_end - begin}
{summary(stats_train)}-------------------------------------------------------
Testing time: {testing_end - training_end}
{summary(stats_test)}
'''))

    env.close()