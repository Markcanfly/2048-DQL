# https://keras.io/examples/rl/deep_q_network_breakout/
# https://towardsdatascience.com/deep-q-learning-tutorial-mindqn-2a4c855abffc

import numpy as np
import tensorflow as tf
import gym_2048
import gym
import random

# Set hyperparameters

# Neuron numbers for the hidden layers
L1 = 256
L2 = 128
L3 = 64
ACTIVATION = 'tanh'
OUTPUT_ACTIVATION = 'softmax'
LEARNING_RATE = 0.7
MIN_EPSILON = 0.01
MAX_EPSILON = 1
EPSILON_DECAY = 0.01

def agent() -> tf.keras.models.Sequential:
    init = tf.keras.initializers.HeUniform()
    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(4,4)),
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

    MIN_REPLAY_SIZE = 200
    if len(replay_memory) < MIN_REPLAY_SIZE:
        return

    batch_size = 64 * 2
    mini_batch = random.sample(replay_memory, batch_size)
    current_states = np.array([transition[0] for transition in mini_batch])
    current_qs_list = model.predict(current_states)
    new_current_states = np.array([transition[3] for transition in mini_batch])
    future_qs_list = target_model.predict(new_current_states)

    X = []
    Y = []
    for index, (observation, action, reward, _, done) in enumerate(mini_batch):
        if not done:
            max_future_q = reward + discount_factor * np.max(future_qs_list[index])
        else:
            max_future_q = reward

        current_qs = current_qs_list[index]
        current_qs[action] = (1 - learning_rate) * current_qs[action] + learning_rate * max_future_q

        X.append(observation)
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

for _ in range(100):
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
            state_tensor = tf.convert_to_tensor(state)
            action_q_values = main_model(state, training=False)
            action = tf.argmax(action_q_values).numpy()

        # TODO gradually decrease epsilon?

        state_next, reward, done, _ = env.step(action)
        replay_memory.append([state, action, reward, state_next, done])
        episode_reward += reward

        if step_count % 5 == 0 or done:
            train(replay_memory, main_model, target_model, done)

        state = state_next

        if done:
            if step_count >= 100:
                # Copy over weights to target model
                target_model.set_weights(main_model.get_weights())
            break

        step_count += 1
    
    episode_count += 1
    print(state)
env.close()