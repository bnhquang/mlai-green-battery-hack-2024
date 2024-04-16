import pandas as pd
from collections import deque
import numpy as np
from policies.policy import Policy

class MARSI(Policy):
    def __init__(self, short_window_size=60, long_window_size=70, rsi_size=10, rsi_thres=10):
        """
        Constructor for the MovingAveragePolicy.

        :param window_size: The number of past market prices to consider for the moving average (default: 5).
        """
        super().__init__()
        self.short_window_size = short_window_size
        self.long_window_size = long_window_size
        self.rsi_size = rsi_size
        self.rsi_thres = rsi_thres
        self.historic_price = deque([45 for i in range(long_window_size+10)], maxlen=long_window_size)
        # historical_data = pd.read_csv('./data/validation_data.csv')
        # self.load_historical(historical_data[:-100])

    """
    
    """

    def act(self, external_state, internal_state):
        market_price = external_state['price']
        self.historic_price.append(market_price)
        price_series = pd.Series(self.historic_price)
        # print(price_series)
        short_ma = price_series.rolling(window=self.short_window_size).mean().iloc[-1]
        long_ma = price_series.rolling(window=self.long_window_size).mean().iloc[-1]
        rsi = self.calculate_rsi(self.historic_price)
        # moving_average = np.mean(self.historic_price)
        diff_percent = abs(abs(short_ma - long_ma) / ((short_ma + long_ma) / 2))
        # print(f'Market price: {market_price}, rsi: {rsi}')
        # print('Diff:', diff_percent)
        if short_ma > long_ma:
            charge_kW = -internal_state['max_charge_rate'] * self.exponential_increase(diff_percent, 8)
            charge_kW = 0 if charge_kW >= self.rsi_thres else charge_kW
            solar_kW_to_battery = external_state['pv_power'] * (1 - self.exponential_increase(diff_percent, 8))
        else:
            charge_kW = internal_state['max_charge_rate'] * self.exponential_increase(diff_percent, 8)
            charge_kW = 0 if charge_kW < self.rsi_thres else charge_kW
            solar_kW_to_battery = external_state['pv_power']

        return solar_kW_to_battery, charge_kW

    def calculate_rsi(self, prices):
        deltas = np.diff(prices)
        gains = deltas[deltas > 0]
        losses = -deltas[deltas < 0]

        avg_gain = np.mean(gains[-self.rsi_size:])
        avg_loss = np.mean(losses[-self.rsi_size:])

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def exponential_increase(self, num, factor):
        return 1 - np.exp(-factor * num)

    # Test with different input values

    def load_historical(self, external_states: pd.DataFrame):
        for price in external_states['price'].values:
            self.historic_price.append(price)


