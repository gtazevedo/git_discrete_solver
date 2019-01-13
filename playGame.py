from collections import deque
import numpy as np
import sys
import gym
sys.path.insert(0, '/home/moony/PycharmProjects/CartPole/CartPole')
import matplotlib.pyplot as plt
from auxFuncs import graph_plotter
from auxFuncs import multicolor_plotter

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Run the env ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def playGame(env, agent, env_max_score, desired_env, learning_rate, max_episodes, state_size, note, sol_score):
    # Variables for plotting
    scores, episodes, action_list, memory_list, desired_score, mean_score, train_done, inverse_train_done = \
        [], [], [], [], [], [], [], []
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

                # Append the values for the graph
                scores.append(score)
                last_100_scores.append(score)
                episodes.append(e)
                #action_list.append(action)
                memory_list.append(len(agent.memory))
                desired_score.append(env_max_score)
                mean_score.append(np.mean(last_100_scores))
                if len(agent.memory) < agent.train_start:
                    train_done.append(0)
                    inverse_train_done.append(1)
                else:
                    train_done.append(1)
                    inverse_train_done.append(0)

                # Every episode, plot the play time
                #graph_plotter(episodes, scores, desired_score, action_list, mean_score, learning_rate, note, desired_env)
                multicolor_plotter(episodes, scores, desired_score, train_done, mean_score, learning_rate, note, desired_env, inverse_train_done)
                # plot_model(agent.model, to_file=('./' + NOTE + ENV_NAME + 'model.png'), show_shapes= True)

                print(" | Episode:", e, " | Score:", score, "/", env_max_score, " | Memory Length:", len(agent.memory),
                      "/", agent.memory.maxlen, " | Epsilon:", round(agent.epsilon,3), " | Reward Given:", reward,
                      " | Mean Score:",  round(np.mean(last_100_scores),3), "/", sol_score,  " |")

                # If the mean of scores of last 100 episodes is the solution score
                # Stop training ~ Commented for now to get more data
                if np.mean(last_100_scores) >= sol_score and agent.train:
                    return
                    #sys.exit()

                # Save the last episodes
                if e > max_episodes - 11:
                    env = gym.wrappers.Monitor(env, "./tmp/" + str(learning_rate) + desired_env, force=True,
                                               video_callable=None, resume=True)

                # Greedy DQN
                if (score >= max_score and agent.train):
                    print("Now we save the better model")
                    max_score = score
                    agent.save_model(
                        "./save_model/" + str(learning_rate) + note + desired_env + "_DDQN18.h5")