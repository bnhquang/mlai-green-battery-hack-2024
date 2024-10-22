import pandas as pd
import numpy as np
from collections import deque
from policies.policy import Policy

class SolarTradingV3(Policy):
    def __init__(self, window_size=282, expo=(52.657237600393806, 16.520328806070264)):
        super().__init__()
        self.window_size = window_size
        self.num_std_dev = 0.2200569382
        self.expo = expo
        self.prices = deque([45 for i in range(window_size)], maxlen=window_size)

    '''
    63.08 window_size=144, num_std_dev=0.5, expo=(12, 1)
    52.34 window_size=72, num_std_dev=0.5, expo=(12, 1)
    62.45 window_size=288, num_std_dev=0.5, expo=(12, 1)
    55.24 window_size=288, num_std_dev=1, expo=(12, 0.5)
    65.62 window_size=288, num_std_dev=0.4, expo=(12, 0.5)
    64.46 window_size=144, num_std_dev=0.4, expo=(12, 0.5)
    64.34 window_size=288, num_std_dev=0.5, expo=(15, 0.4)
    64.75 window_size=250, num_std_dev=0.1, expo=(15, 0.4)
    65.61 window_size=288, num_std_dev=0.4, expo=(15, 0.3)
    66.56 window_size=275, num_std_dev=0.1, expo=(15, 0.4)
    67.02 window_size=274, num_std_dev=0.1, expo=(15, 1)
    window_size=274, num_std_dev=0.1, expo=(30, 1)
    '''
    
    def update_num_std_dev(self):
        # Calculate a simple volatility index (standard deviation of price changes)
        price_series = pd.Series(self.prices)
        price_changes = price_series.diff().dropna()
        volatility_index = price_changes.std()
        threshold_high = 100
        threshold_low = 15

        # Adjust num_std_dev based on the volatility index
        if volatility_index > threshold_high:
            self.num_std_dev = 0.25  # More volatile market
            self.window_size = 230
        elif volatility_index < threshold_low:
            self.num_std_dev = 0.05  # Less volatile market
            self.window_size = 320
        else:
            self.num_std_dev = 0.2200569382  # Moderate volatility
            self.window_size = 282



    def act(self, external_state, internal_state):
        current_price = external_state['price']
        self.update_num_std_dev()
        pv_power = external_state['pv_power']
        max_charge_rate = internal_state['max_charge_rate']

        self.prices.append(current_price)
        price_series = pd.Series(self.prices)
        avg = price_series.rolling(window=self.window_size).mean().iloc[-1]
        std_dev = np.std(self.prices)
        upper_band = avg + (std_dev * self.num_std_dev)
        lower_band = avg - (std_dev * self.num_std_dev)

        diff_percent = abs(abs(current_price - avg) / ((avg + current_price) / 2))
        # print(diff_percent)

        if current_price > upper_band:
            charge_kW = -max_charge_rate * self.exponential_increase(diff_percent, self.expo[0])
            solar_kW_to_battery = pv_power * (1 - self.exponential_increase(diff_percent, self.expo[0]))
        elif current_price < lower_band:
            charge_kW = max_charge_rate * self.exponential_increase(diff_percent, self.expo[1])
            solar_kW_to_battery = pv_power
        else:
            charge_kW = 0
            solar_kW_to_battery = pv_power


        return solar_kW_to_battery, charge_kW

    def exponential_increase(self, num, factor):
        """
        Helper function to compute an exponential increase based on the difference percentage.

        :param num: The difference percentage.
        :param factor: The factor to influence the growth rate.
        :return: The exponential growth result.
        """
        return 1 - np.exp(-factor * num)

    def load_historical(self, external_states):
        for price in external_states['price'].values:
            self.prices.append(price)
