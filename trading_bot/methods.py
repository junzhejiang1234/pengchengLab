import os
import logging

import numpy as np

from tqdm import tqdm

from .utils import format_currency, format_position
from .ops import get_state
from .ops import form_state
import matplotlib.pyplot as plt
import empyrical

# new version with newly defined reward
# aim to optimize daily level transaction
# use opportunity cost as reward
# buy all sell all


def train_model(revenue, initial_revenue, agent, episode, data, ep_count=50, batch_size=32, window_size=10):
    total_profit = 0
    data_length = len(data) - 1

    agent.inventory = []
    avg_loss = []

    state = get_state(data, 0, window_size + 1)

    def get_c(t):
        t = max(0, t)
        t = min(t, data_length)
        return data[t, 3]

    # initial revenue equals value of portfolio equals cash plus stock
    cash = initial_revenue
    stock = 0.0
    transaction = 0.0003
    # print(state)
    state = form_state(
        state, cash / initial_revenue, stock * 0 / initial_revenue
    )  # append the percentage of cash and stock in portfolio

    hold_penalty = 0.003
    advance = 1

    for t in tqdm(range(data_length), total=data_length, leave=True, desc="Episode {}/{}".format(episode, ep_count)):
        # if (t + 1) in [1900, 3854, 5808, 7762, 9716, 11725,13734,15743,17752,19618,21627,23636,25645,27654,29663,31672,33681,35690,37699,39708,41717,43535,45544,47498,49507,51516,53525,55391,57257,59123,61132,63141,64959,66968] and t < data_length - 1:
        if (t + 5) % 3019 in [0, 1, 2, 3, 4]:
            # ready to move on to next stock
            if stock > 0:
                bought_price = agent.inventory.pop(0)
                delta = (get_c(t) * (1 - transaction) - bought_price) * stock
                total_profit += delta
            cash += stock * get_c(t) * (1 - transaction)
            stock = 0
            revenue.append(initial_revenue)
            initial_revenue = cash
            # prepare to feed to next round
            state = get_state(data, t + 1, window_size + 1)  # state of next stock
            state = form_state(state, cash / initial_revenue, stock * 0 / initial_revenue)
            continue
        revenue.append(initial_revenue)
        reward = 0
        next_state = get_state(data, t + 1, window_size + 1)

        # select an action
        action = agent.act(state)

        # buy and sell would be rewarded by
        # gain/loss incurred by buy or gain/loss avoided by sell
        # advance means how much advance we should look ahead when evaluating reward, e.g. when advance = 1, we'll use cps(t+1) / cps(t)

        # BUY
        if action == 0:
            if cash > 0:
                # if buy, use all available cash
                agent.inventory.append(get_c(t) * (1 + transaction))
                stock += cash / (get_c(t) * (1 + transaction))
                cash = 0.0
                old_revenue = initial_revenue
                initial_revenue = cash + stock * get_c(t)
                # calculate reward using opportunity cost
                buy_gain = stock * (get_c(t + advance) - get_c(t)) - (
                    stock * get_c(t) * transaction
                )  # portfolio gain minus transaction cost
                hold_gain = 0
                sell_gain = 0
                reward = buy_gain - max(hold_gain, sell_gain)
                reward /= old_revenue
                reward = buy_gain / initial_revenue  # (buy_gain - sell_gain) / old_revenue # new way 0430 mark
            else:
                old_revenue = initial_revenue
                initial_revenue = cash + stock * get_c(t)
                # old reward
                # reward = (
                #     data[min(t + 1, data_length), 4] / data[t, 4]
                #     - 1
                #     - transaction
                #     - max(0, 1 - data[min(t + 1, data_length), 4] / data[t, 4] - transaction)
                # )
                # reward *= 0.5

                # change to hold reward
                # action = 0
                buy_gain = (
                    cash * (get_c(t + advance) / get_c(t) - 1) * (1 - transaction)
                    - cash * transaction
                    + stock * (get_c(t + advance) - get_c(t))
                )  # portfolio gain minus transaction cost
                sell_gain = -stock * get_c(t) * transaction
                hold_gain = stock * (get_c(t + advance) - get_c(t))
                reward = hold_gain - max(sell_gain, buy_gain)
                reward /= old_revenue
                reward = buy_gain / initial_revenue  # (buy_gain - sell_gain) / old_revenue # new way 0430 mark
                # reward -= hold_penalty
                # add small reward for making the decision not to hold
                # reward += 0.3 * (get_c(t + 1) / get_c(t) - 1 - transaction)
            # if reward < 0:
            #     reward *= 2.1

        # SELL
        elif action == 1:
            if stock > 0:
                # if sell, sell all available stocks
                bought_price = agent.inventory.pop(0)
                delta = (get_c(t) * (1 - transaction) - bought_price) * stock
                total_profit += delta
                cash += (get_c(t) * (1 - transaction)) * stock
                old_stock = stock
                stock = 0.0
                old_revenue = initial_revenue
                initial_revenue = cash + stock * get_c(t)
                # calculate reward using opportunity cost
                buy_gain = old_stock * (get_c(t + advance) - get_c(t))
                hold_gain = old_stock * (get_c(t + advance) - get_c(t))
                sell_gain = -old_stock * get_c(t) * transaction  # minus transaction fee
                reward = sell_gain - max(hold_gain, buy_gain)
                reward /= old_revenue
                reward = (sell_gain - buy_gain) / old_revenue  # new way 0430 mark
                reward = -transaction - (buy_gain) / initial_revenue
            else:
                old_revenue = initial_revenue
                initial_revenue = cash + stock * get_c(t)
                # old reward
                # reward = (
                #     1
                #     - data[min(t + 1, data_length), 4] / data[t, 4]
                #     - transaction
                #     - max(0, data[min(t + 1, data_length), 4] / data[t, 4] - 1 - transaction)
                # )
                # reward *= 0.5

                # change to hold reward
                # action = 0
                buy_gain = (
                    cash * (get_c(t + advance) / get_c(t) - 1) * (1 - transaction)
                    - cash * transaction
                    + stock * (get_c(t + advance) - get_c(t))
                )  # portfolio gain minus transaction cost
                sell_gain = -stock * get_c(t) * transaction
                hold_gain = stock * (get_c(t + advance) - get_c(t))
                reward = hold_gain - max(sell_gain, buy_gain)
                reward /= old_revenue
                reward = (sell_gain - buy_gain) / old_revenue  # new way 0430 mark
                # reward -= 0.3 * hold_penalty
                # add small reward for making the decision not to hold
                # reward += 0.5 * (1 - get_c(t + 1) / get_c(t) - transaction)
                reward = -buy_gain / initial_revenue

        # HOLD
        else:
            print("wtf!!!")  # means there should be no other actions
            # old_revenue = initial_revenue
            # initial_revenue = cash + stock * get_c(t)
            # buy_gain = (
            #     cash * (get_c(t + 1) / get_c(t) - 1) * (1 - transaction)
            #     - cash * transaction
            #     + stock * (get_c(t + 1) - get_c(t))
            # )  # portfolio gain minus transaction cost
            # sell_gain = -stock * get_c(t) * transaction
            # hold_gain = stock * (get_c(t + 1) - get_c(t))
            # reward = hold_gain - max(sell_gain, buy_gain)
            # reward /= old_revenue
            # reward -= hold_penalty
            # pass

        reward = np.log((1 + reward)) * 100
        if action == 0 and reward < -3.5:
            # reward -= 3
            reward = 1 / (3 * 3.5 * 3.5) * np.power(reward, 3) + (
                -3.5 + 3.5 / 3
            )  # non linear penalty for negative loss incurred by buy action
        next_state = form_state(next_state, cash / initial_revenue, stock * get_c(t) / initial_revenue)
        done = t == data_length - 1
        agent.remember(state, action, reward, next_state, done)

        if (len(agent.memory) > 15 * batch_size and len(agent.memory) > 0.5 * agent.memory.maxlen) and ((episode * data_length + t) % (30) == 0):
            loss = agent.train_experience_replay(batch_size)
            avg_loss.append(loss)

        state = next_state

    if episode % 1 == 0:
        agent.save(episode)

    daily_return = np.array(revenue)[1:] / np.array(revenue)[:-1] - 1
    draw_back = empyrical.max_drawdown(daily_return)
    sharpe = empyrical.sharpe_ratio(daily_return)
    annual_ret = empyrical.annual_return(daily_return)
    print("stock left {}".format(stock))
    print("max_draw_down {}".format(draw_back))
    print("sharpe_ratio {}".format(sharpe))
    print("annual return {}".format(annual_ret))

    print("final value: ", revenue[len(revenue) - 1])
    return (episode, ep_count, total_profit, np.mean(np.array(avg_loss)))


# new version comes with holding return
def evaluate_model(revenue, initial_revenue, agent, data, window_size, debug):
    total_profit = 0
    data_length = len(data) - 1

    def get_c(t):
        t = max(0, t)
        t = min(t, data_length)
        return data[t, 3]

    history = []
    agent.inventory = []
    daily_return = []
    state = get_state(data, 0, window_size + 1)
    # initial revenue equals value of portfolio equals cash plus stock
    cash = initial_revenue
    stock = 0
    transaction = 0.0003
    state = form_state(state, cash / initial_revenue, stock * 0 / initial_revenue)

    for t in range(data_length):
        revenue.append(initial_revenue)
        # reward = 0
        next_state = get_state(data, t + 1, window_size + 1)

        # select an action
        action = agent.act(state, is_eval=True)

        # BUY
        if action == 0:
            if cash > 0:
                agent.inventory.append(get_c(t) * (1 + transaction))  # change from 1.00001 to 1.00015
                stock += cash / (get_c(t) * (1 + transaction))
                cash = 0
                initial_revenue = cash + stock * get_c(t)
                history.append((get_c(t), "BUY"))
                if debug:
                    logging.debug("Buy at: {}".format(format_currency(get_c(t))))

                daily_return.append(initial_revenue / revenue[-1] - 1)
            else:
                initial_revenue = cash + stock * get_c(t)
                daily_return.append(initial_revenue / revenue[-1] - 1)
                history.append((get_c(t), "INTENTION TO BUY"))

        # SELL
        elif action == 1:
            if stock > 0:
                bought_price = agent.inventory.pop(0)
                delta = (get_c(t) * (1 - transaction) - bought_price) * stock
                total_profit += delta
                cash += get_c(t) * (1 - transaction) * stock
                stock = 0
                initial_revenue = cash + stock * get_c(t)
                history.append((get_c(t), "SELL"))
                if debug:
                    logging.debug(
                        "Sell at: {} | Position: {}".format(
                            format_currency(get_c(t)), format_position(get_c(t) - bought_price)
                        )
                    )
                daily_return.append(initial_revenue / revenue[-1] - 1)
            else:
                initial_revenue = cash + stock * get_c(t)
                daily_return.append(initial_revenue / revenue[-1] - 1)
                history.append((get_c(t), "INTENTION TO SELL"))
        # HOLD
        else:
            initial_revenue = stock * get_c(t) + cash
            # daily_return.append(0.0)
            history.append((get_c(t), "HOLD"))
            daily_return.append(initial_revenue / revenue[-1] - 1)

        next_state = form_state(next_state, cash / initial_revenue, stock * get_c(t) / initial_revenue)
        done = t == data_length - 1
        # agent.memory.append((state, action, reward, next_state, done))

        state = next_state
        if done:

            revenue = np.array(revenue)
            daily_return = np.array(daily_return)
            plt.plot(revenue)
            plt.show()
            if len(daily_return) > 60:
                daily_return_season = daily_return[:60]
                draw_back = empyrical.max_drawdown(daily_return_season)
                sharpe = empyrical.sharpe_ratio(daily_return_season)
                annual_ret = empyrical.annual_return(daily_return_season)
                print("max_draw_down 1st season {}".format(draw_back))
                print("sharpe_ratio 1st season {}".format(sharpe))
                print("annual return 1st season {}".format(annual_ret))
            if len(daily_return) > 120:
                daily_return_halfy = daily_return[:120]
                draw_back = empyrical.max_drawdown(daily_return_halfy)
                sharpe = empyrical.sharpe_ratio(daily_return_halfy)
                annual_ret = empyrical.annual_return(daily_return_halfy)
                print("max_draw_down 1st half year {}".format(draw_back))
                print("sharpe_ratio 1st half year {}".format(sharpe))
                print("annual return 1st half year {}".format(annual_ret))

            draw_back = empyrical.max_drawdown(daily_return)
            sharpe = empyrical.sharpe_ratio(daily_return)
            annual_ret = empyrical.annual_return(daily_return)
            print("stock left {}".format(stock))
            print("max_draw_down {}".format(draw_back))
            print("sharpe_ratio {}".format(sharpe))
            print("annual return {}".format(annual_ret))
            return total_profit, history
