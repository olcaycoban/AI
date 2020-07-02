import gym
import random
import numpy as np
from statistics import median, mean
from collections import Counter

from keras.layers import Dropout,Dense,Activation,Flatten
from keras.models import Sequential
from keras import optimizers
import keras

LR = 1e-3
env = gym.make("CartPole-v0")
env.reset()
goal_steps = 500
score_requirement = 50
initial_games = 10000


def some_random_game_first():
    for episode in range(15):
        env.reset()
        for t in range(goal_steps):
            env.render()
            action=env.action_space.sample()
            obervation,reward,done,info=env.step(action)
            if done:
                break


def initial_population():
    training_data = []
    scores = []
    accepted_scores = []
    for _ in range(initial_games):
        score = 0
        game_memory = []
        prev_observation = []
        for _ in range(goal_steps):
            action = random.randrange(0, 2)
            observation, reward, done, info = env.step(action)

            if len(prev_observation) > 0:
                game_memory.append([prev_observation, action])
            prev_observation = observation
            score += reward
            if done: break

        if score >= score_requirement:
            accepted_scores.append(score)
            for data in game_memory:
                if data[1] == 1:
                    output = [0, 1]
                elif data[1] == 0:
                    output = [1, 0]

                training_data.append([data[0], output])

        env.reset()
        scores.append(score)

    training_data_save = np.array(training_data)
    np.save('saved.npy', training_data_save)

    print('Average accepted score:', mean(accepted_scores))
    print('Median score for accepted scores:', median(accepted_scores))
    print(Counter(accepted_scores))

    return training_data

def neural_network_model(input_size):
    model = Sequential()
    model.add(Dense(256, input_shape=(input_size,1), activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(2, activation='softmax'))


    opt = optimizers.SGD(learning_rate=0.05)
    model.compile(
        optimizer=opt,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


def train_model(training_data, model=False):
    X = np.array([i[0] for i in training_data])
    X=X.reshape(-1, len(training_data[0][0]),1)
    y = [i[1] for i in training_data]

    hist = model.fit(X, y, epochs=15, batch_size=32)
    model.evaluate(X, y)[1]

training_data=initial_population()
model=neural_network_model(4)
train_model(training_data,model)
