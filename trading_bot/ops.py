import os
import math
import logging

import numpy as np


def sigmoid(x):
    """Performs sigmoid operation"""
    try:
        if x < 0:
            return 1 - 1 / (1 + math.exp(x))
        return 1 / (1 + math.exp(-x))
    except Exception as err:
        print("Error in sigmoid: " + err)


def get_state(data, t, n_days):
    """Returns an n-day state representation ending at time t"""
    d = t - n_days + 1
    # block = data[d : t + 1] if d >= 0 else -d * [data[0]] + data[0 : t + 1]  # pad with t0

    ## zy change
    # block = 0
    # if d >= 0:
    #     block = data[d : t + 1]
    # else:
    #     block = data[0]
    #     for i in range(-d):
    #         if
    #      -d * [data[0]] + data[0 : t + 1]  # pad with t0

    return np.array([data[t]])

    res = []
    for i in range(n_days - 1):
        res.append(sigmoid(block[i + 1] - block[i]))
    return np.array([res])


def form_state(state, cash, stock):
    """ Returns state variable with cash and stock"""
    old_state = list(state[0])
    # change closing price in state to log level
    # old_state[3] = int(np.log(old_state[3])) - 4
    # what if we let go of state[3]
    # old_state[3] = 0
    # nah let's keep it in a different way
    old_state[3] = int(np.log(old_state[3]) - 5)
    old_state = (
        old_state[:3] + old_state[4:-21]
    )  # means to only keep stock related data, get rid of vix data, also get rid of close price data (old state [3])
    new_state = np.array(old_state)
    return np.array([new_state])
