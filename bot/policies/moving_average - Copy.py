import pandas as pd
from collections import deque
import numpy as np
from policies.policy import Policy

class VerySimplePolicy(Policy):
    def __init__(self, short_window_size=5, long_window_size=10, historical_price_len=15):
        """
        Constructor for the MovingAveragePolicy.

        :param window_size: The number of past market prices to consider for the moving average (default: 5).
        """
        super().__init__()
        self.short = deque([45 for i in range(short_window_size)], maxlen=short_window_size)
        self.long = deque([45 for i in range(long_window_size)], maxlen=long_window_size)
        # self.historic_price = deque([45 for i in range(historical_price_len)], maxlen=historical_price_len)
        # historical_data = pd.read_csv('./data/validation_data.csv')
        # self.load_historical(historical_data[:20])

# 2.79 6-12 with short percent



    def act(self, external_state, internal_state):
        market_price = external_state['price']
        # prev_short_ma = np.mean(self.short)
        self.short.append(market_price)
        self.long.append(market_price)
        short_ma = np.mean(self.short)
        long_ma = np.mean(self.long)
        # print(f'Max: {max_moving_average}, Min: {min_moving_average}, Average: {moving_average}')
        # moving_average = np.mean(self.historic_price)
        diff_percent = abs(short_ma - market_price) / (abs(short_ma) + abs(market_price))
        # print(f'Market price: {market_price}, ma: {short_ma}, prev ma: {prev_short_ma}')
        # print('Diff:', diff_percent)
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