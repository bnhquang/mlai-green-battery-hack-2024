import pandas as pd
from collections import deque
import numpy as np
import pandas_ta as ta
import torch

from policies.policy import Policy
from DQN.DQN_Agent import DQN_Agent

class DQN_Policy(Policy):
    def __init__(self, **kwargs):
        """
        Constructor for the Policy class. It can take flexible parameters.
        Contestants are free to maintain internal state and use any auxiliary data or information
        within this class.
        """
        super().__init__()
        self.historical_df = None
        self.agent = DQN_Agent(
            state_size=7,
            replay_mem_size=50000,
            gamma=0.99,
            epsilon=1,
            batch_size=128,
            n_actions=2,
            min_eps=0.05,
            lr=0.0001,
            eps_dec=0.00001
        )
        self.agent.load_agent('./DQN/checkpoints/dqn_50.zip')
        self.actions = [0, 1]



    def act(self, external_state, internal_state):
        """
        Method to be called when the policy needs to make a decision.

        :param external_state: A dictionary containing the current market data.
        :param internal_state: A dictionary containing the internal state of the policy.
        :return: A tuple (amount to route from solar panel to battery: int, amount to charge battery from grid: int). Note: any solar power not routed to the battery goes directly to the grid and you earn the current spot price.
        """
        external_state_df = pd.DataFrame([external_state])
        self.historical_df = pd.concat([self.historical_df, external_state_df], ignore_index=True)
        preprocessed_data = self.preprocess_data(self.historical_df)
        current_state = preprocessed_data.iloc[-1].values
        # print(current_state)
        t_current_state = torch.tensor(current_state, dtype=torch.float32)
        action_idx = self.agent.choose_action(t_current_state)
        buy = bool(self.actions[action_idx])
        total_solar_kW = external_state['pv_power']

        short_ma = preprocessed_data.iloc[-1]['SMA_10']
        long_ma = preprocessed_data.iloc[-1]['SMA_288']
        # print(short_ma, long_ma)

        diff_percent = abs(abs(short_ma - long_ma) / ((short_ma + long_ma) / 2)) if (short_ma and long_ma != 0) else 1
        if buy is True:
            charge_kW = internal_state['max_charge_rate'] * self.exponential_increase(diff_percent, 5)
            solar_kW_to_battery = total_solar_kW
        else:
            charge_kW = -internal_state['max_charge_rate'] * self.exponential_increase(diff_percent, 15)
            solar_kW_to_battery = 0

        # actions = [
        #     [1, 1],         # charge full, keep all pv_power
        #     [-1, 0],       # discharge full, keep no pv_power
        #     [0.5, 0.5],     # charge half, keep half pv_power
        #     [-0.5, 0],        # discharge half, keep no pv_power
        # ]
        # external_state = self.preprocess_external_state(external_state)
        # # print(external_state)
        # current_state = tuple(external_state.values) + tuple(internal_state.values()) + \
        #                 (13 - internal_state['battery_soc'],)
        # current_state_t = torch.Tensor(current_state)
        # action_idx = self.agent.choose_action(current_state_t)
        # grid_charge_percent, pv_charge_percent = actions[action_idx]
        # charge_kW = internal_state['max_charge_rate'] * grid_charge_percent
        # total_solar_kW = external_state['pv_power']
        # solar_kW_to_battery = total_solar_kW * pv_charge_percent


        return (solar_kW_to_battery, charge_kW)

    def exponential_increase(self, num, factor):
        return 1 - np.exp(-factor * num)

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
        df = df.astype(float).fillna(0)
        return df

    def load_historical(self, external_states):
        self.historical_df = external_states
