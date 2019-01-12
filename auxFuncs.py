import matplotlib.pyplot as plt
import numpy as np


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


