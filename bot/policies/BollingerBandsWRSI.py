import pandas as pd
import numpy as np
from collections import deque
from policies.policy import Policy

class SolarTradingV4(Policy):
    def __init__(self, window_size=282, expo=(52.657237600393806, 16.520328806070264), num_std_dev=0.22005693825218395):
        super().__init__()
        self.window_size = window_size
        self.num_std_dev = num_std_dev
        self.expo = expo
        self.rsi_period = 3
        self.prices = deque([45 for i in range(window_size)], maxlen=window_size)



    
    def calculate_rsi(self):
        # Convert prices to Series to utilize rolling window functions
        price_series = pd.Series(list(self.prices))
        delta = price_series.diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)

        avg_gain = gain.rolling(window=self.rsi_period, min_periods=self.rsi_period).mean().iloc[-1]
        avg_loss = loss.rolling(window=self.rsi_period, min_periods=self.rsi_period).mean().iloc[-1]

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    



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
        
        rsi = self.calculate_rsi()

        if rsi > 70 or current_price > upper_band:
            charge_kW = -max_charge_rate * self.exponential_increase(diff_percent, self.expo[0])
            solar_kW_to_battery = pv_power * (1 - self.exponential_increase(diff_percent, self.expo[0]))
        elif rsi < 30 or current_price < lower_band:
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
