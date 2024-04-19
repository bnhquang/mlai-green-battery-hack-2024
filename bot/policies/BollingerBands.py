import pandas as pd
import numpy as np
from collections import deque
from policies.policy import Policy

class BollingerBandsPolicy(Policy):
    def __init__(self, window_size=144, num_std_dev=2):
        super().__init__()
        self.window_size = window_size
        self.num_std_dev = num_std_dev
        self.prices = deque([45 for i in range(window_size)], maxlen=window_size)

    def act(self, external_state, internal_state):
        current_price = external_state['price']
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
            charge_kW = -max_charge_rate * self.exponential_increase(diff_percent, 8)
            solar_kW_to_battery = pv_power * (1 - self.exponential_increase(diff_percent, 8))
        elif current_price < lower_band:
            charge_kW = max_charge_rate * self.exponential_increase(diff_percent, 1)
            solar_kW_to_battery = pv_power * self.exponential_increase(diff_percent, 1)
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