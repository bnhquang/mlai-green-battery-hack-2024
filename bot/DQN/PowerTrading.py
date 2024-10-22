import time
import random
import numpy as np
from environment import BatteryEnv
from environment import kWh_to_kW
import pandas as pd
import pandas_ta as ta

class PowerTrading():
    def __init__(self, battery_env: BatteryEnv):
        self.preprocessed_data = self.preprocess_data(battery_env.market_data)
        # print(self.preprocessed_data.dtypes)
        self.battery_env = battery_env
        self.battery_cap = battery_env.battery.capacity_kWh
        self.external_state, self.internal_state = self.battery_env.initial_state()

        # self.actions = [
        #     [1, 1],         # charge full, keep all pv_power
        #     [-1, 0],       # discharge full, keep no pv_power
        #     [0.5, 0.5],     # charge half, keep half pv_power
        #     [-0.5, 0],        # discharge half, keep no pv_power
        # ]

        self.actions = [0, 1]

        # Observation space: 3 float values representing current price, pv power, cap remain (cap - power), current balance
        # Action space: char full, charge half, discharge full, discharge half
        self.action_space = [i for i in range(len(self.actions))]

    def step(self, action_idx):
        reward = self.make_action(action_idx)
        state = self.preprocessed_data.iloc[self.battery_env.current_step].values
        # print(len(state))
        return state, reward

    def make_action(self, action_idx):
        # print("Action:", action_idx)
        reward = 0
        battery_charge_prev = self.internal_state['battery_soc']
        price_prev = self.external_state['price']
        buy = bool(self.actions[action_idx])
        total_solar_kW = self.external_state['pv_power']
        short_ma = self.preprocessed_data.iloc[self.battery_env.current_step]['SMA_10']
        long_ma = self.preprocessed_data.iloc[self.battery_env.current_step]['SMA_288']
        # print(short_ma, long_ma)

        diff_percent = abs(abs(short_ma - long_ma) / ((short_ma + long_ma) / 2)) if (short_ma and long_ma != 0) else 1

        if buy is True:
            charge_kW = self.internal_state['max_charge_rate'] * diff_percent
            solar_kW_to_battery = total_solar_kW
        else:
            charge_kW = -self.internal_state['max_charge_rate'] * diff_percent
            solar_kW_to_battery = 0
        self.external_state, self.internal_state = self.battery_env.step(charge_kW, solar_kW_to_battery, total_solar_kW)
        reward = self.internal_state['profit_delta']

        # charge_kWh = (self.battery_cap - self.internal_state[
        #     'battery_soc']) * grid_charge_percent if grid_charge_percent >= 0 \
        #     else self.internal_state['battery_soc'] * grid_charge_percent
        # charge_kW = self.internal_state['max_charge_rate'] * grid_charge_percent
        # # print(charge_kW)
        # total_solar_kW = self.external_state['pv_power']
        # solar_kW_to_battery = total_solar_kW * pv_charge_percent
        # temp_external_state, self.internal_state = self.battery_env.step(charge_kW, solar_kW_to_battery, total_solar_kW)
        # if temp_external_state is not None:
        #     self.external_state = temp_external_state
        #
        # battery_delta = self.internal_state['battery_soc'] - battery_charge_prev
        # profit_reward = self.internal_state['profit_delta'] * 5
        # charge_reward = battery_delta / price_prev if price_prev > 0 else battery_delta * 10 if price_prev < 0 else 0
        # discharge_reward = -battery_delta * price_prev * 5
        # capped_penalty = -10 if (self.internal_state['battery_soc'] == self.battery_env.battery.capacity_kWh or
        #                          self.internal_state['battery_soc'] == 0) else 0
        # reward += profit_reward + charge_reward + discharge_reward + capped_penalty
        return reward

    def reset(self):
        self.battery_env.battery.reset()
        self.battery_env.current_step = 0

    def preprocess_data(self, market_data):
        df = market_data
        df = df.loc[:, ['timestamp', 'price']]
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['hour_sin'] = df['timestamp'].apply(lambda x: np.sin(2 * np.pi * (x.hour + x.minute / 60) / 24))
        df['hour_cos'] = df['timestamp'].apply(lambda x: np.cos(2 * np.pi * (x.hour + x.minute / 60) / 24))
        df.dropna(inplace=True)
        df.drop('timestamp', axis=1, inplace=True)
        df['SMA_10'] = df['price'].rolling(window=10).mean()
        df['SMA_144'] = df['price'].rolling(window=144).mean()
        df['SMA_288'] = df['price'].rolling(window=288).mean()
        df['RSI'] = ta.rsi(df['price'], length=14)
        df.fillna(0, inplace=True)
        return df

    # def charge_quantity(self, percent):
    #     charge_amount = (self.battery_cap - self.battery_power) * percent
    #     # print(f'Charge amount: {charge_amount}, pv_power: {self.pv_power}')
    #     # quantity = min(charge_amount // self.pv_power, self.balance // self.price) if self.price >= 0 else \
    #     #     charge_amount // self.pv_power
    #     quantity = min(charge_amount / self.pv_power, self.balance / self.price) if self.price >= 0 else \
    #         charge_amount / self.pv_power
    #     return quantity
    #
    # def discharge_quantity(self, percent):
    #     discharge_amount = self.battery_power * percent
    #     # print(f'Discharge amount: {discharge_amount}, pv_power: {self.pv_power}')
    #     # quantity = discharge_amount // self.pv_power
    #     quantity = (discharge_amount / self.pv_power) if self.price >= 0 else \
    #         min(discharge_amount / self.pv_power, self.balance / abs(self.price))
    #     return quantity


