# https://keras.io/examples/rl/deep_q_network_breakout/
# https://towardsdatascience.com/deep-q-learning-tutorial-mindqn-2a4c855abffc

import numpy as np
import tensorflow as tf
import gym_2048
import gym
import random
from evaluate import largest_tile, summary
from tqdm import trange

import textwrap
from datetime import datetime, timedelta

# Set hyperparameters
N_EPISODES = 50
N_TEST = 100

# Neuron numbers for the hidden layers
L1 = 256
L2 = 512
L3 = 64
ACTIVATION = 'tanh'
OUTPUT_ACTIVATION = 'softmax'
LEARNING_RATE = 0.85
MIN_EPSILON = 0.01
MAX_EPSILON = 1
EPSILON_DECAY = 0.01
BATCH_SIZE = 64

begin = datetime.now()

def agent() -> tf.keras.models.Sequential:
    init = tf.keras.initializers.HeUniform()
    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(None, 16)),
        tf.keras.layers.Dense(L1, ACTIVATION, kernel_initializer=init),
        tf.keras.layers.Dropout(.2),
        tf.keras.layers.Dense(L2, ACTIVATION, kernel_initializer=init),
        tf.keras.layers.Dense(L3, ACTIVATION, kernel_initializer=init),
        tf.keras.layers.Dense(4, OUTPUT_ACTIVATION)
    ])
    model.compile(loss=tf.keras.losses.Huber(), optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE), metrics=['accuracy'])
    return model

def train(replay_memory, model, target_model, done):
    learning_rate = LEARNING_RATE
    discount_factor = 0.618

    MIN_REPLAY_SIZE = 128
    if len(replay_memory) < MIN_REPLAY_SIZE:
        return

    batch_size = BATCH_SIZE
    mini_batch = random.sample(replay_memory, batch_size)
    current_states = np.array([tf.convert_to_tensor(transition[0].reshape(1, 16)) for transition in mini_batch])
    current_qs_list = model.predict_on_batch(current_states)
    new_current_states = np.array([tf.convert_to_tensor(transition[3].reshape(1, 16)) for transition in mini_batch])
    future_qs_list = target_model.predict_on_batch(new_current_states)

    X = []
    Y = []
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
main_model = agent()
# Update every nth step
target_model = agent()

episode_count = 0
epsilon = MAX_EPSILON # copy from hyperparam

replay_memory = [] # TODO Replace with faster dtype (deque?)

stats_train = {}

for episode in trange(N_EPISODES):
    state = env.reset()
    episode_reward = 0
    done = False
    step_count = 0
    while not done:
        # Decide next action
        if random.random() < epsilon: # Exploration
            action = random.choice((0,1,2,3))
        else: # Exploitation
            # Use Q-values
            state_tensor = state.reshape([1, 16])
            action_q_values = main_model(state_tensor, training=False)
            action = np.argmax(action_q_values)

        epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * np.exp(-EPSILON_DECAY * episode)

        state_next, reward, done, _ = env.step(action)
        replay_memory.append([state, action, reward, state_next, done])
        episode_reward += reward

        if step_count % 5 == 0 or done:
            train(replay_memory, main_model, target_model, done)

        state = state_next

        max_tile = largest_tile(state)

        if done:
            # Add to stats
            stats_train[max_tile] = stats_train.get(max_tile, 0) + 1
            if step_count >= 100:
                # Copy over weights to target model
                target_model.set_weights(main_model.get_weights())
            break
        
        step_count += 1
    
    episode_count += 1

training_end = datetime.now()
print(summary(stats_train))

stats_test = {}
# Test
for episode in range(N_TEST):
    state = env.reset()
    done = False
    step_count = 0
    while not done:
        state_tensor = state.reshape([1, 16])
        action_q_values = main_model(state_tensor, training=False)
        action = np.argmax(action_q_values)
        previous_state = state
        state, _, done, _ = env.step(action)

        # Explore if state doesn't change from move
        if np.array_equal(state, previous_state):
            state, _, done, _ = env.step(random.choice((0,1,2,3)))

        step_count += 1

        if done:
            # Add to stats
            max_tile = largest_tile(state)
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