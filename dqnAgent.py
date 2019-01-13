from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential
from collections import deque
import numpy as np
import random


class DoubleDQNAgent:
    def __init__(self, state_size, action_size, learning_rate, hidden_layer, hidden_layer2):
        # If the network is already trained, set to False
        self.train = True
        # If its desired to see the env animation, set to True
        self.render = False
        # Get size of state and action
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_layer = hidden_layer
        self.hidden_layer2 = hidden_layer2

        # These are hyper parameters for the Double DQN
        self.discount_factor = 0.99  # also known as gamma
        self.learning_rate = learning_rate  # 0.001 for Adam, 0.002 if Nadam or Adamax
        if (self.train):
            self.epsilon = 1.0  # if we are training, exploration rate is set to max
        else:
            self.epsilon = 1e-6  # if we are just running, exploration rate is min
        self.epsilon_decay = 0.999
        self.epsilon_min = 1e-6
        self.batch_size = 64
        self.train_start = action_size ** action_size * 250  # when will we start training, maybe some higher numbers. Default is 1000
        # Create replay memory using deque
        self.memory = deque(maxlen=int(action_size ** action_size * 750 ))  # 2000 is default. Try some higher numbers

        # Create main model and target model
        self.model = self.build_model()
        self.target_model = self.build_model()

        # Initialize target model so that the parameters of model and target model are the same
        self.update_target_model()

    # Aproximate Q function using Neural Network
    # State is input and Q value of each action is output of the network
    def build_model(self):
        model = Sequential()
        model.add(Dense(self.hidden_layer, input_dim=self.state_size, activation='relu',
                        kernel_initializer='he_uniform'))
        model.add(Dense(self.hidden_layer2, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(self.hidden_layer, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size, activation='linear', kernel_initializer='he_uniform'))
        model.summary()
        optimizer = Adam(lr=self.learning_rate)  # try Adamax, Adam and Nadam
        model.compile(loss='mse', optimizer=optimizer)
        return model

    # Updates the target model to be the same as the model
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    # Get action from model using epsilon-greedy policy
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model.predict(state)
            return np.argmax(q_value[0])

    # Save sample <s, a, r, s'> to the replay memory
    def replay_memory(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    # Pick random samples from the replay memory
    def train_model(self):
        if len(self.memory) < self.train_start:
            # If you don't have enough samples, just return
            return

        batch_size = min(self.batch_size, len(self.memory))
        # Takes random samples from the memory to train on
        minibatch = random.sample(self.memory, batch_size)

        # Now we do the experience replay
        state, action, reward, next_state, is_terminal = zip(*minibatch)
        state = np.concatenate(state)
        next_state = np.concatenate(next_state)

        target = self.model.predict(state)
        target_next = self.model.predict(state)
        target_val = self.target_model.predict(next_state)

        # Like Q learning, get maximum Q value at s' but from target model
        for i in range(batch_size):
            if is_terminal[i]:
                target[i][action[i]] = reward[i]
            else:
                # The key point of Double DQN selection of action is from the model
                # The update is from the target model
                a = np.argmax(target_next[i])
                target[i][action[i]] = reward[i] + self.discount_factor * (target_val[i][a])

        # Choose the option to use the train_on_batch or fit function, should be the same
        # self.model.fit(state, target, batch_size=self.batch_size, epochs=1, verbose=0)
        self.model.train_on_batch(state, target)

    # Load the saved model
    def load_model(self, name):
        self.model.load_weights(name)

    # Save the model which is under training
    def save_model(self, name):
        self.model.save_weights(name)
