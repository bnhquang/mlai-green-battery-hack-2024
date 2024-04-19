import pandas as pd
import numpy as np
from collections import deque
from policies.policy import Policy

class BollingerBandsPolicy(Policy):
    def __init__(self, window_size=20, num_std_dev=2):
        super().__init__()
        self.window_size = window_size
        self.num_std_dev = num_std_dev
        self.prices = deque(maxlen=window_size)

    def act(self, external_state, internal_state):
        current_price = external_state['price']
        pv_power = external_state['pv_power']
        max_charge_rate = internal_state['max_charge_rate']

        self.prices.append(current_price)
        if len(self.prices) >= self.window_size:
            avg = np.mean(self.prices)
            std_dev = np.std(self.prices)
            upper_band = avg + (std_dev * self.num_std_dev)
            lower_band = avg - (std_dev * self.num_std_dev)
            
            diff_percent = abs(current_price - avg) / avg

            if current_price > upper_band:
                charge_kW = -max_charge_rate * self.exponential_increase(diff_percent, 8)
                charge_kW = 0 if charge_kW > -0.5 else charge_kW
                solar_kW_to_battery = pv_power * (1 - self.exponential_increase(diff_percent, 8))
                
            elif current_price < lower_band:
                charge_kW = max_charge_rate * self.exponential_increase(diff_percent, 8)
                charge_kW = 0 if charge_kW < 2 else charge_kW
                solar_kW_to_battery = pv_power * self.exponential_increase(diff_percent, 8)
                
            else:
                solar_kW_to_battery = pv_power / 2
                charge_kW = min(max_charge_rate, pv_power / 2)
        else:
            # default action
            solar_kW_to_battery = pv_power / 2
            charge_from_grid = 0

        return solar_kW_to_battery, charge_from_grid

    def exponential_increase(self, num, factor):
        """
        Helper function to compute an exponential increase based on the difference percentage.

        :param num: The difference percentage.
        :param factor: The factor to influence the growth rate.
        :return: The exponential growth result.
        """
        return 1 - np.exp(-factor * num)

    def load_historical(self, external_states):
        for state in external_states:
            self.prices.append(state['price'])
