from auxFuncs import plotData
from collections import deque
import numpy as np
import sys
import gym

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Run the env ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def playGame(env, agent, env_max_score, desired_env, learning_rate, max_episodes, state_size, note):
    # Variables for plotting
    scores, episodes, action_list, memory_list, marker, desired_score, mean_score = [], [], [], [], [], [], []
    last_100_scores = deque(maxlen= 100)
    # Variable for saving the model
    max_score = -999

    # If we are not training aka we are running, load the model
    if (not agent.train):
        print("Now we load the saved model")
        agent.load_model("./save_model/" + str(learning_rate) + note + desired_env + "_DDQN18.h5")

    # If we are training:
    for e in range(max_episodes):
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
            elif done and score >= (env_max_score - 1):
                reward += 500
            else:
                reward += -500

            # Clip all positive rewards at 1 and all negative rewards at -1, leaving 0 rewards unchanged
            reward = np.clip(reward, -1, 1)  # Trying this

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

                plotData(scores, last_100_scores, episodes, action_list, memory_list, marker, desired_score, mean_score,
                         score, e, action, agent.memory, env_max_score, learning_rate, note, desired_env):

                # plot_model(agent.model, to_file=('./' + NOTE + ENV_NAME + 'model.png'), show_shapes= True)

                print(" | Episode:", e, " | Score:", score, " | Memory Length:", len(agent.memory), "/",
                      agent.memory.maxlen,
                      " | Epsilon:", agent.epsilon, " | Reward Given:", reward, " | Env Max Score:",
                      env_max_score,
                      " | Training starts in:", agent.train_start, " |")

                # If the mean of scores of last 10 episode is bigger than 490
                # Stop training ~ Commented for now to get more data
                if np.mean(scores[-min(10, len(scores)):]) == env_max_score and agent.train:
                    sys.exit()

                # Save the last episodes
                if e > max_episodes - 11:
                    env = gym.wrappers.Monitor(env, "./tmp/" + desired_env, force=True,
                                               video_callable=None, resume=True)

                # Greedy DQN
                if (score >= max_score and agent.train):
                    print("Now we save the better model")
                    max_score = score
                    agent.save_model(
                        "./save_model/" + str(learning_rate) + note + desired_env + "_DDQN18.h5")