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

            if current_price > upper_band:
                solar_kW_to_battery = 0
                charge_from_grid = 0
                
            elif current_price < lower_band:
                solar_kW_to_battery = pv_power
                charge_from_grid = min(max_charge_rate, pv_power)
                
            else:
                solar_kW_to_battery = pv_power / 2
                charge_from_grid = min(max_charge_rate, pv_power / 2)
        else:
            # default action
            solar_kW_to_battery = pv_power / 2
            charge_from_grid = 0

        return solar_kW_to_battery, charge_from_grid

    def load_historical(self, external_states):
        for state in external_states:
            self.prices.append(state['price'])
