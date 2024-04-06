import random
from policies.policy import Policy

class RandomPolicy(Policy):
    def __init__(self, charge_probability=0.5):
        """
        Constructor for the RandomPolicy.

        :param charge_probability: The probability of charging the battery (default: 0.5).
        """
        super().__init__()
        self.charge_probability = charge_probability

    def act(self, external_state, internal_state):
        """
        Select a random action based on the charge probability.

        :param state: A dictionary containing market data.
        :param info: A dictionary containing additional information relevant to decision-making.
        :return: quantity (kW) to charge/discharge for the current step.
        """
        if random.random() < self.charge_probability:
            # Charge the battery at the maximum rate
            return 0, internal_state['max_charge_rate']
        else:
            # Discharge the battery at the maximum rate
            return 0, -internal_state['max_charge_rate']
    
    def load_historical(self, external_state):
        pass