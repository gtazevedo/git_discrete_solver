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

from dqnAgent import DoubleDQNAgent
from playGame import playGame
from auxFuncs import env_max_score
import gym
import tensorflow as tf


if __name__ == "__main__":
    # Tensorflow GPU optimization
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    from keras import backend as K

    K.set_session(sess)

    # Creates an argparse to get the env name needs to add parser for train or not
    # parser = argparse.ArgumentParser(description='Gives args for the code')
    # parser.add_argument('-e', '--env', help='Tells what env to run', required=True)
    # parser.add_argument('-lr', '--learning_rate', help='Tells the lr for the env', required=True)
    # args = vars(parser.parse_args())

    # Run in loop:
    max_episodes = 3000
    hidden_layer = 50  # Original was 24 and 50 has a good score
    hidden_layer2 = 50
    # list of all desired envs
    #env_col = ['CartPole-v0', 'CartPole-v1', 'MountainCar-v0', 'Acrobot-v1', 'LunarLander-v2']
    desired_env = 'CartPole-v0'
    # list of all desired lr
    #lr_col = [0.0001, 5e-5, 0.001, 0.005, 0.0005, 1e-5]
    lr_col = [0.005, 0.0005]
    #for desired_env in env_col:
    for learning_rate in lr_col:
        # In case of CartPole-v1, you can play until 500 time step
        # env = gym.make(args['env'])
        env = gym.make(desired_env)
        # Get size of state and action from environment
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n
        # gets the env max score
        # max_score = env_max_score(args['env'])
        max_score = env_max_score(desired_env)

        # agent = DoubleDQNAgent(state_size, action_size, args['learning_rate'])
        agent = DoubleDQNAgent(state_size, action_size, learning_rate, hidden_layer, hidden_layer2)

        playGame(env, agent, max_score, desired_env, learning_rate, max_episodes, state_size)