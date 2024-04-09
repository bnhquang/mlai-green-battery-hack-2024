import pandas as pd
from collections import deque
# import numpy as np
import torch as torch

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
        self.agent = DQN_Agent(
            replay_mem_size=50000,
            gamma=0.99,
            epsilon=1,
            batch_size=128,
            n_actions=4,
            min_eps=0.05,
            lr=0.0001,
            eps_dec=0.00001
        )
        self.agent.load_agent('./bot/DQN/checkpoints/dqn_2.zip')

    def act(self, external_state, internal_state):

        """
        Method to be called when the policy needs to make a decision.

        :param external_state: A dictionary containing the current market data.
        :param internal_state: A dictionary containing the internal state of the policy.
        :return: A tuple (amount to route from solar panel to battery: int, amount to charge battery from grid: int). Note: any solar power not routed to the battery goes directly to the grid and you earn the current spot price.
        """
        actions = [
            [1, 1],         # charge full, keep all pv_power
            [-1, 0],       # discharge full, keep no pv_power
            [0.5, 0.5],     # charge half, keep half pv_power
            [-0.5, 0],        # discharge half, keep no pv_power
        ]
        external_state = self.preprocess_external_state(external_state)
        # print(external_state)
        current_state = tuple(external_state.values) + tuple(internal_state.values()) + \
                        (13 - internal_state['battery_soc'],)
        current_state_t = torch.Tensor(current_state)
        action_idx = self.agent.choose_action(current_state_t)
        grid_charge_percent, pv_charge_percent = actions[action_idx]
        charge_kW = internal_state['max_charge_rate'] * grid_charge_percent
        total_solar_kW = external_state['pv_power']
        solar_kW_to_battery = total_solar_kW * pv_charge_percent
        return (solar_kW_to_battery, charge_kW)

    def preprocess_external_state(self, external_state):
        external_state['timestamp'] = pd.to_datetime(external_state['timestamp']).timestamp()
        return external_state

    def load_historical(self, external_states):
        """
        Load historical data to the policy. This method is called once before the simulation starts.

        :param external_states: A list of dictionaries containing historical market data to be used as relevant context
        when acting later.
        """
        pass
