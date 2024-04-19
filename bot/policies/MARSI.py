import pandas as pd
from collections import deque
import numpy as np
from policies.policy import Policy

class MARSI(Policy):
    def __init__(self, short_window_size=10, long_window_size=288, rsi_size=14, rsi_thres=(50, 60), expo=(8, 0.5)):
        """
        Constructor for the MovingAveragePolicy.

        :param window_size: The number of past market prices to consider for the moving average (default: 5).
        """
        super().__init__()
        self.short_window_size = short_window_size
        self.long_window_size = long_window_size
        self.rsi_size = rsi_size
        self.rsi_thres = rsi_thres
        self.expo = expo
        self.historic_price = deque([45 for i in range(long_window_size+10)], maxlen=long_window_size)
        # historical_data = pd.read_csv('./data/validation_data.csv')
        # self.load_historical(historical_data[:-100])

    """
    28.47 short_window_size=20, long_window_size=180, rsi_size=14, rsi_thres=(35, 60), expo=(9, 3)
    27.29 short_window_size=30, long_window_size=180, rsi_size=14, rsi_thres=(35, 60), expo=(9, 3)
    28.30 short_window_size=30, long_window_size=288, rsi_size=14, rsi_thres=(35, 60), expo=(9, 3)
    30.32 short_window_size=30, long_window_size=288, rsi_size=144, rsi_thres=(40, 60), expo=(9, 3)
    28.95 short_window_size=40, long_window_size=288, rsi_size=144, rsi_thres=(45, 60), expo=(10, 3)
    20.73 short_window_size=70, long_window_size=288, rsi_size=144, rsi_thres=(35, 55), expo=(9, 3)
    31.78 short_window_size=25, long_window_size=288, rsi_size=144, rsi_thres=(40, 60), expo=(9, 3)
    32.23 (60.37) short_window_size=20, long_window_size=288, rsi_size=144, rsi_thres=(40, 60), expo=(9, 3)
    31.18 short_window_size=20, long_window_size=350, rsi_size=144, rsi_thres=(40, 60), expo=(9, 3)
    31.67 (58.86) short_window_size=20, long_window_size=400, rsi_size=144, rsi_thres=(40, 60), expo=(9, 3)
    60.37 short_window_size=15, long_window_size=288, rsi_size=144, rsi_thres=(40, 60), expo=(9, 3)
    61.11 short_window_size=10, long_window_size=288, rsi_size=144, rsi_thres=(40, 60), expo=(9, 3)
    61.34 short_window_size=10, long_window_size=288, rsi_size=144, rsi_thres=(40, 60), expo=(10, 2)
    61.62 short_window_size=10, long_window_size=288, rsi_size=144, rsi_thres=(40, 60), expo=(11, 1)
    61.56 short_window_size=10, long_window_size=288, rsi_size=144, rsi_thres=(40, 60), expo=(13, 2)
    61.26 short_window_size=5, long_window_size=288, rsi_size=144, rsi_thres=(40, 60), expo=(9, 2)
    63.18 short_window_size=10, long_window_size=288, rsi_size=144, rsi_thres=(40, 60), expo=(9, 0.9)
    64.06 short_window_size=10, long_window_size=288, rsi_size=144, rsi_thres=(40, 60), expo=(8, 0.9)
    63.31 short_window_size=10, long_window_size=288, rsi_size=144, rsi_thres=(40, 60), expo=(6, 0.9)
    64.26 short_window_size=10, long_window_size=288, rsi_size=144, rsi_thres=(40, 60), expo=(8, 0.5)
    61.21 short_window_size=10, long_window_size=288, rsi_size=14, rsi_thres=(40, 30), expo=(8, 0.5)
    64.38 short_window_size=10, long_window_size=288, rsi_size=14, rsi_thres=(40, 50), expo=(8, 0.5)
    short_window_size=10, long_window_size=288, rsi_size=14, rsi_thres=(50, 60), expo=(8, 0.5)
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
            charge_kW = -internal_state['max_charge_rate'] * self.exponential_increase(diff_percent, self.expo[0])
            charge_kW = charge_kW if rsi >= self.rsi_thres[0] else 0
            solar_kW_to_battery = external_state['pv_power'] * (1 - self.exponential_increase(diff_percent, self.expo[0]))
        else:
            charge_kW = internal_state['max_charge_rate'] * self.exponential_increase(diff_percent, self.expo[1])
            charge_kW = charge_kW if rsi < self.rsi_thres[1] else 0
            solar_kW_to_battery = external_state['pv_power']
        # print(charge_kW)
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


