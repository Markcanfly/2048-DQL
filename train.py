# https://keras.io/examples/rl/deep_q_network_breakout/
# https://towardsdatascience.com/deep-q-learning-tutorial-mindqn-2a4c855abffc

import numpy as np
import tensorflow as tf
import gym_2048
import gym

# Set hyperparameters

# Neuron numbers for the hidden layers
L1 = 256
L2 = 128
L3 = 64
ACTIVATION = 'tanh'
OUTPUT_ACTIVATION = 'softmax'
LEARNING_RATE = 0.001
def agent() -> tf.keras.models.Sequential:
    init = tf.keras.initializers.HeUniform()
    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(4,)),
        tf.keras.layers.Dense(L1, ACTIVATION, kernel_initializer=init),
        tf.keras.layers.Dropout(),
        tf.keras.layers.Dense(L2, ACTIVATION, kernel_initializer=init),
        tf.keras.layers.Dense(L3, ACTIVATION, kernel_initializer=init),
        tf.keras.layers.Dense(4, OUTPUT_ACTIVATION)
    ])
    model.compile(loss=tf.keras.losses.Huber(), optimizer=tf.keras.optimizers.Adam(lr=LEARNING_RATE), metrics=['accuracy'])
    return model


env = gym.make('2048-v0')
env.seed(42)

env.reset()
env.render()

