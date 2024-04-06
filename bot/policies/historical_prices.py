import json
import pandas as pd
import numpy as np
from policies.policy import Policy

class HistoricalPricePolicy(Policy):
    def __init__(self):
        super().__init__()

        # note, this only works because the file in question exists in that directory 
        # (relative to the root of the project) and is tracked by github. The Dockerfile
        # makes sure the entire contents of bot are copied into the docker container which
        # gets run for real-time trading, so since `example_historical.json` is in the bot
        # directory, it will be available to the policy at submission
        with open('bot/data/example_historical.json') as f:
            historical_data = json.load(f)

        historical_prices = []

        for price in historical_data['historical_prices']:
            historical_prices.append(price['price'])
        
        self.historical_mean = np.mean(historical_prices)
        
        self.buy_price = self.historical_mean * 0.9
        self.sell_price = self.historical_mean * 1.1

    def act(self, external_state, internal_state):
        market_price = external_state['price']

        if market_price >= self.buy_price:
            charge_kW = internal_state['max_charge_rate']
        elif market_price <= self.sell_price:
            charge_kW = -internal_state['max_charge_rate']

        return 0, charge_kW

    def load_historical(self, external_states: pd.DataFrame):
        pass
