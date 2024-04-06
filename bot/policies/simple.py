from policies.policy import Policy

class SimplePolicy(Policy):
    def __init__(self, charge_kW=10):
        self.charge_kW = charge_kW

    def act(self, external_state, internal_state):
        return 0, self.charge_kW

    def load_historical(self, external_states):
        pass
