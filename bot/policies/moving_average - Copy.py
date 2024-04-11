import pandas as pd
from collections import deque
import numpy as np
from policies.policy import Policy

class VerySimplePolicy(Policy):
    def __init__(self, short_window_size=8, long_window_size=16):
        """
        Constructor for the MovingAveragePolicy.

        :param window_size: The number of past market prices to consider for the moving average (default: 5).
        """
        super().__init__()
        self.short = deque([45, 45, 45], maxlen=short_window_size)
        self.long = deque([45, 45, 45, 45, 45], maxlen=long_window_size)
        # historical_data = pd.read_csv('./data/validation_data.csv')
        # self.load_historical(historical_data[:10])

# 6-6-6
# Average profit ($): 203.44 ± 96.61
# Average profit inc rundown ($): 343.25

    def act(self, external_state, internal_state):
        market_price = external_state['price']
        self.short.append(market_price)
        self.long.append(market_price)
        short_ma = np.mean(self.short)
        long_ma = np.mean(self.long)
        # print(f'Max: {max_moving_average}, Min: {min_moving_average}, Average: {moving_average}')
        diff_percent = abs(short_ma - long_ma) / max(short_ma, long_ma)
        if short_ma > long_ma:
            charge_kW = -internal_state['max_charge_rate'] * diff_percent
            solar_kW_to_battery = 0
        else:
            charge_kW = internal_state['max_charge_rate'] * diff_percent
            solar_kW_to_battery = external_state['pv_power']

        return solar_kW_to_battery, charge_kW

    def load_historical(self, external_states: pd.DataFrame):
        for price in external_states['price'].values:
            self.short.append(price)
            self.long.append(price)