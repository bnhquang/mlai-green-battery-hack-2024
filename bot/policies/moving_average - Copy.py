import pandas as pd
from collections import deque
import numpy as np
from policies.policy import Policy

class VerySimplePolicy(Policy):
    def __init__(self, short_window_size=60, long_window_size=70, historical_price_len=15):
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
# 2.59 5-10 with short percent
# 3.08 8-16 with short
# 3.97 10-20 with long
# 4.95 10-20 diff * 2
# 4.56 10-20 expo 5
# 5.61 15-30 expo 5
# 8.06 20-40 expo 2
# 9.22 30-40 expo 5
# 10.69 40-50 expo 6
# 11.75 50-60 expo 7
# 12.80 60-70 expo 8
# 12.59 80-90 expo 10
# 11.24 90-100 expo 9
# 12.88 60-70 expo 7-7
# 13.15 60-70 expo 8-6
# 14.61 60-70 expo 8-5
    def act(self, external_state, internal_state):
        market_price = external_state['price']
        # prev_short_ma = np.mean(self.short)
        self.short.append(market_price)
        self.long.append(market_price)
        short_ma = np.mean(self.short)
        long_ma = np.mean(self.long)
        # print(f'Max: {max_moving_average}, Min: {min_moving_average}, Average: {moving_average}')
        # moving_average = np.mean(self.historic_price)
        diff_percent = abs(abs(short_ma - long_ma) / ((short_ma + long_ma) / 2))
        # print(f'Market price: {market_price}, ma: {short_ma}, prev ma: {prev_short_ma}')
        # print('Diff:', diff_percent)
        if short_ma > long_ma:
            charge_kW = -internal_state['max_charge_rate'] * self.exponential_increase(diff_percent, 8)
            solar_kW_to_battery = external_state['pv_power'] * (1 - self.exponential_increase(diff_percent, 8))
        else:
            charge_kW = internal_state['max_charge_rate'] * self.exponential_increase(diff_percent, 4)
            solar_kW_to_battery = external_state['pv_power']

        return solar_kW_to_battery, charge_kW

    def exponential_increase(self, num, factor):
        return 1 - np.exp(-factor * num)

    # Test with different input values

    def load_historical(self, external_states: pd.DataFrame):
        for price in external_states['price'].values:
            self.short.append(price)
            self.long.append(price)

