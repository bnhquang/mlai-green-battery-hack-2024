import pandas as pd
from collections import deque
import numpy as np
from policies.policy import Policy

class VerySimplePolicy(Policy):
    def __init__(self, window_size=6):
        """
        Constructor for the MovingAveragePolicy.

        :param window_size: The number of past market prices to consider for the moving average (default: 5).
        """
        super().__init__()
        # print("Hey")
        self.window_size = window_size
        self.max_price_history = deque(maxlen=(window_size))
        self.min_price_history = deque(maxlen=(window_size))
        self.price_history = deque(maxlen=window_size)

        self.max_price_history.append(0)
        self.min_price_history.append(100)
        self.price_history.append(0)

# 6-6-6
# Average profit ($): 203.44 ± 96.61
# Average profit inc rundown ($): 343.25

    def act(self, external_state, internal_state):
        market_price = external_state['price']
        max_moving_average = np.mean(self.max_price_history)
        min_moving_average = np.mean(self.min_price_history)
        moving_average = np.mean(self.price_history)
        # print(f'Max: {max_moving_average}, Min: {min_moving_average}, Average: {moving_average}')

        if max_moving_average < market_price:
            self.max_price_history.append(market_price)
            # print(np.mean(self.max_price_history))
        if market_price < min_moving_average:
            self.min_price_history.append(market_price)
            # print(self.min_price_history)
        self.price_history.append(market_price)

        if market_price > max_moving_average:
            charge_kW = -internal_state['max_charge_rate']
            solar_kW_to_battery = 0
        elif market_price < min_moving_average:
            charge_kW = internal_state['max_charge_rate']
            solar_kW_to_battery = external_state['pv_power']
        elif market_price > moving_average:
            # charge_kW = -internal_state['max_charge_rate'] / 2
            charge_kW = 0
            solar_kW_to_battery = 0
            self.max_price_history.append(max_moving_average * 0.9)
        else:
            # charge_kW = internal_state['max_charge_rate'] / 2
            charge_kW = 0
            solar_kW_to_battery = external_state['pv_power']
            self.min_price_history.append(min_moving_average * 0.9)


        return solar_kW_to_battery, charge_kW

    def load_historical(self, external_states: pd.DataFrame):   
        for price in external_states['price'].values:
            self.price_history.append(price)