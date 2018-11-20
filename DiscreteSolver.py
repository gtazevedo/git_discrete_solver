# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ References: ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# https://github.com/yanpanlau/CartPole/blob/master/doubleDQN/CartPole_DQN.py
# https://github.com/rlcode/reinforcement-learning/blob/master/2-cartpole/2-double-dqn/cartpole_ddqn.py
# https://github.com/openai/gym/wiki/CartPole-v0
# https://github.com/openai/gym/wiki/MountainCar-v0
# https://github.com/tokb23/dqn/blob/master/ddqn.py
# https://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/
# https://matplotlib.org/tutorials/introductory/pyplot.html#sphx-glr-tutorials-introductory-pyplot-py
# https://matplotlib.org/tutorials/introductory/sample_plots.html
# https://keras.io/visualization/
# https://medium.com/@gabogarza/deep-reinforcement-learning-policy-gradients-8f6df70404e6
# https://medium.com/mlreview/speeding-up-dqn-on-pytorch-solving-pong-in-30-minutes-81a1bd2dff55
# https://allan.reyes.sh/projects/gt-rl-lunar-lander/
# http://rrusin.blogspot.com/2017/04/landing-on-moon-using-ai-bot.html <- look into this code
# https://github.com/yanpanlau/DDPG-Keras-Torcs <- try using this code maybe
# https://yanpanlau.github.io/2016/10/11/Torcs-Keras.html
# https://pt.slideshare.net/AakashChotrani/deep-q-learning-with-lunar-lander
# https://github.com/Seraphli/YADQN/blob/master/code/openai/LunarLander-v2/Experiment_5/evaluation.py <- Great code reference for monitor
# https://medium.freecodecamp.org/improvements-in-deep-q-learning-dueling-double-dqn-prioritized-experience-replay-and-fixed-58b130cc5682

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ To do: ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Plot policy
# Plot decision boundary
# args to choose stuff after is done


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Imports: ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import sys
import gym
import random
import numpy as np
from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
from keras.optimizers import Adamax
from keras.optimizers import Nadam
from keras.models import Sequential
from keras import metrics
import matplotlib.pyplot as plt
from keras.utils import plot_model
import tensorflow as tf
from sysconfig import get_path

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Notes: ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Needs to implement argparse and organize the code in different files after dev

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Setup: ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
EPISODES = 3000
HIDDEN_LAYER = 50 # Original was 24 and 50 has a good score
HIDDEN_LAYER2 = 50
ENV_NAME = "LunarLander-v2" # CartPole-v0 | CartPole-v1 | MountainCar-v0 | LunarLander-v2 | Acrobot-v1
NOTE = "lr_0001_gamma_999_Reward_test_Batch_Opt_Adam_2HL_50_50_Reward_Clipped_withBonus_Ep_3k_Replay_actio_times_40k_Train_action_times_5k"
# Next tests are ep 2k replay 10k train 5k adamax lr 0.002 | ep 2k replay 10k train 1k adam lr 0.001
# | ep 2k replay 10k train 1k adamax lr 0.002

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Double DQN ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class DoubleDQNAgent:
    def __init__(self, state_size, action_size):
        # If the network is already trained, set to False
        self.train = True
        # If its desired to see the env animation, set to True
        self.render = False
        # Get size of state and action
        self.state_size = state_size
        self.action_size = action_size

        # These are hyper parameters for the Double DQN
        self.discount_factor = 0.999  # also known as gamma
        self.learning_rate = 0.0001  # 0.001 for Adam, 0.002 if Nadam or Adamax
        if (self.train):
            self.epsilon = 1.0  # if we are training, exploration rate is set to max
        else:
            self.epsilon = 1e-6  # if we are just running, exploration rate is min
        self.epsilon_decay = 0.999
        self.epsilon_min = 1e-6
        self.batch_size = 64
        self.train_start = action_size* 5000  # when will we start training, maybe some higher numbers. Default is 1000
        # Create replay memory using deque
        self.memory = deque(maxlen=action_size * 40000)  # 2000 is default. Try some higher numbers

        # Create main model and target model
        self.model = self.build_model()
        self.target_model = self.build_model()

        # Initialize target model so that the parameters of model and target model are the same
        self.update_target_model()

    # Aproximate Q function using Neural Network
    # State is input and Q value of each action is output of the network
    def build_model(self):
        model = Sequential()
        model.add(Dense(HIDDEN_LAYER, input_dim=self.state_size, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(HIDDEN_LAYER, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(HIDDEN_LAYER2, activation = 'relu', kernel_initializer='he_uniform'))
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
        #self.model.fit(state, target, batch_size=self.batch_size, epochs=1, verbose=0)
        self.model.train_on_batch(state, target)

    # Load the saved model
    def load_model(self, name):
        self.model.load_weights(name)

    # Save the model which is under training
    def save_model(self, name):
        self.model.save_weights(name)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Run the env ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def playGame(env, agent, env_max_score):
    # Variables for plotting
    scores, episodes, action_list, memory_list, marker, desired_score = [], [], [], [], [], []
    # Variable for saving the model
    max_score = -999

    # If we are not training aka we are running, load the model
    if (not agent.train):
        print("Now we load the saved model")
        agent.load_model("./save_model/" + NOTE + ENV_NAME + "_DDQN18.h5")

    # If we are training:
    for e in range(EPISODES):
        done = False
        score = 0
        state = env.reset()
        state = np.reshape(state, [1, state_size])

        while not done:
            # If render is set to true, show the user the env. This makes the process slower.
            if agent.render:
                env.render()

            # get action for the current state and go one step in environment
            action = agent.get_action(state)
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])

            # If an action make the episode end, then give it penalty of -1
            # Score for this is max number of steps -1

            # try this instead of the above to see if we have any improvs
            if not done:
                reward = reward
            elif done and score >= (env_max_score-1):
                reward += 500
            else:
                reward += -500

            # Clip all positive rewards at 1 and all negative rewards at -1, leaving 0 rewards unchanged
            reward= np.clip(reward, -1, 1) #Trying this

            # Save the sample <s, a, r, s'> to the replay memory
            agent.replay_memory(state, action, reward, next_state, done)
            # Every time step do the training if we are training
            if (agent.train):
                agent.train_model()
            score += reward
            state = next_state

            if done:
                # Resets the env for new episode
                env.reset()
                # Every episode update the target model to be same with model
                agent.update_target_model()

                # Every episode, plot the play time
                scores.append(score)
                episodes.append(e)
                action_list.append(action)
                memory_list.append(len(agent.memory))
                marker.append(5000)
                desired_score.append(env_max_score)

                fig = plt.figure() # creates the figure
                ax1 = fig.add_subplot(3, 1, 1) # creates the subplots
                ax2 = fig.add_subplot(3, 1, 2)
                ax3 = fig.add_subplot(3, 1, 3)

                ax1.plot(episodes, scores, 'b-', lw = 0.5) # plots the agent score
                ax1.plot(episodes, desired_score, 'g-', lw=0.4)
                ax2.scatter(episodes, action_list, facecolor = 'blue', cmap = plt.cm.get_cmap("winter"), alpha= 0.15)
                ax3.scatter(episodes, scores, facecolor = 'blue', alpha = 0.15)

                ax1.set_title('Episode score')
                ax2.set_title('Last action taken by episode')
                ax3.set_title('Episode Score')

                fig.savefig("./save_graph/" + NOTE + ENV_NAME + "_DDQN18.png")
                plt.close(fig)

                #plot_model(agent.model, to_file=('./' + NOTE + ENV_NAME + 'model.png'), show_shapes= True)

                print(" | Episode:", e, " | Score:", score, " | Memory Length:", len(agent.memory), "/", agent.memory.maxlen,
                      " | Epsilon:", agent.epsilon, " | Reward Given:", reward, " | Env Max Score:", env_max_score,
                      " | Training starts in:", agent.train_start, " |")

                # If the mean of scores of last 10 episode is bigger than 490
                # Stop training ~ Commented for now to get more data
                #if np.mean(scores[-min(10, len(scores)):]) == env_max_score and agent.train:
                    #sys.exit()

                # Save the last episodes
                if e > EPISODES - 11:
                    env = gym.wrappers.Monitor(env, "./tmp/" + ENV_NAME, force=True, video_callable = None, resume = True)

                # Greedy DQN
                if (score >= max_score and agent.train):
                    print("Now we save the better model")
                    max_score = score
                    agent.save_model("./save_model/" + NOTE + ENV_NAME + "_DDQN18.h5")



def env_max_score(ENV_NAME):
    if ENV_NAME == "CartPole-v0":
        return 200
    elif ENV_NAME == "CartPole-v1":
        return 500
    elif ENV_NAME == "MountainCar-v0":
        return -110
    elif ENV_NAME == "LunarLander-v2":
        return 200
    elif ENV_NAME == "Acrobot-v1":
        return -80

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Main Func  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == "__main__":
    # Tensorflow GPU optimization
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    from keras import backend as K
    K.set_session(sess)

    # In case of CartPole-v1, you can play until 500 time step
    env = gym.make(ENV_NAME)
    # Get size of state and action from environment
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    max_score = env_max_score(ENV_NAME)

    agent = DoubleDQNAgent(state_size, action_size)

    playGame(env, agent, max_score)
