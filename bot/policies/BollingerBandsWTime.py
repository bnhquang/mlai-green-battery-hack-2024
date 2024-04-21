import pandas as pd
import numpy as np
from collections import deque
from policies.policy import Policy

class SolarTrading2(Policy):
    def __init__(self, window_size_norm=274, num_std_dev_norm=0.1, expo_norm=(30, 1),
                 window_size_peak=274, num_std_dev_peak=0.1, expo_peak=(30, 1)):
        super().__init__()
        # Normal parameters
        self.window_size_norm = window_size_norm
        self.num_std_dev_norm = num_std_dev_norm
        self.expo_norm = expo_norm
        self.prices_norm = deque([45 for _ in range(window_size_norm)], maxlen=window_size_norm)
        
        # Peak parameters
        self.window_size_peak = window_size_peak
        self.num_std_dev_peak = num_std_dev_peak
        self.expo_peak = expo_peak
        self.prices_peak = deque([45 for _ in range(window_size_peak)], maxlen=window_size_peak)

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


    def act(self, external_state, internal_state):
        current_price = external_state['price']
        current_time = pd.to_datetime(external_state['timestamp']).time()
        pv_power = external_state['pv_power']
        max_charge_rate = internal_state['max_charge_rate']
        
        # Determine if it's peak hours
        is_peak = current_time.hour >= 8 and current_time.hour <= 15

        if is_peak:
            window_size = self.window_size_peak
            num_std_dev = self.num_std_dev_peak
            expo = self.expo_peak
            prices = self.prices_peak
        else:
            window_size = self.window_size_norm
            num_std_dev = self.num_std_dev_norm
            expo = self.expo_norm
            prices = self.prices_norm

        prices.append(current_price)
        price_series = pd.Series(list(prices))
        avg = price_series.rolling(window=window_size).mean().iloc[-1]
        std_dev = np.std(list(prices))
        upper_band = avg + (std_dev * num_std_dev)
        lower_band = avg - (std_dev * num_std_dev)

        diff_percent = abs(abs(current_price - avg) / ((avg + current_price) / 2))

        if current_price > upper_band:
            charge_kW = -max_charge_rate * self.exponential_increase(diff_percent, expo[0])
            solar_kW_to_battery = pv_power * (1 - self.exponential_increase(diff_percent, expo[0]))
        elif current_price < lower_band:
            charge_kW = max_charge_rate * self.exponential_increase(diff_percent, expo[1])
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
            self.prices_norm.append(price)
            self.prices_peak.append(price)
