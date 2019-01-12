import matplotlib.pyplot as plt

def plotData(scores, last_100_scores, episodes, action_list, memory_list, marker, desired_score, mean_score, score, e,
             action, memory, env_max_score, learning_rate, NOTE, desired_env):
    # Every episode, plot the play time
    scores.append(score)
    last_100_scores.append(score)
    episodes.append(e)
    action_list.append(action)
    memory_list.append(len(memory))
    marker.append(5000)
    desired_score.append(env_max_score)
    mean_score.append(np.mean(last_100_scores))

    fig = plt.figure()  # creates the figure
    ax1 = fig.add_subplot(3, 1, 1)  # creates the subplots
    ax2 = fig.add_subplot(3, 1, 2)
    ax3 = fig.add_subplot(3, 1, 3)
    # ax4 = fig.add_subplot(4,1,4)

    ax1.plot(episodes, scores, 'b-', lw=0.5)  # plots the agent score
    ax1.plot(episodes, desired_score, 'r-', lw=0.4)  # plots the desired score in each env
    ax2.scatter(episodes, action_list, facecolor='blue', cmap=plt.cm.get_cmap("winter"),
                alpha=0.15)  # plots the last action taken in each env
    ax3.scatter(episodes, scores, facecolor='blue', alpha=0.15)  # plots the agent score
    ax3.plot(episodes, mean_score, 'r-', lw=0.4)  # plots the mean score of the last 100 eposides

    ax1.title.set_text('Episode score')
    ax2.title.set_text('Last action taken by episode')
    ax3.title.set_text('Episode Score')
    # ax4.title.set_text('Last 100 episodes mean score')

    # Makes sure theres no overlap
    plt.tight_layout()

    fig.savefig("./save_graph/" + str(learning_rate) + NOTE + desired_env + "_DDQN18.png")
    plt.close(fig)

def env_max_score(env_name):
    if env_name == "CartPole-v0":
        return 200
    elif env_name == "CartPole-v1":
        return 500
    elif env_name == "MountainCar-v0":
        return -110
    elif env_name == "LunarLander-v2":
        return 200
    elif env_name == "Acrobot-v1":
        return -80