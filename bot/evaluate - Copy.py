import time
import argparse
import os
import random
import pandas as pd
from datetime import datetime
import numpy as np
import json

from policies import policy_classes
from environmentcopy import BatteryEnv, PRICE_KEY, TIMESTAMP_KEY
from plotting import plot_results
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK


print("Current Working Directory:", os.getcwd())


def float_or_none(value):
    if value.lower() == 'none':
        return None
    try:
        return float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value} is not a float or 'None'")


def load_config(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)['policy']

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def run_down_battery(battery_environment: BatteryEnv, market_prices):
    last_day_prices = market_prices[-288:]
    assumed_rundown_price = np.mean(last_day_prices)
    rundown_profits = []
    
    while battery_environment.battery.state_of_charge_kWh > 0:
        kWh_removed = battery_environment.battery.discharge_at(battery_environment.battery.max_charge_rate_kW)
        rundown_profits.append(battery_environment.kWh_to_profit(kWh_removed, assumed_rundown_price))

    return rundown_profits

def run_trial(battery_environment: BatteryEnv, policy):
    profits, socs, market_prices, battery_actions, solar_actions, pv_inputs, timestamps = [], [], [], [], [], [], []

    external_state, internal_state = battery_environment.initial_state()
    while True:
        pv_power = float(external_state["pv_power"])
        solar_kW_to_battery, charge_kW = policy.act(external_state, internal_state)

        market_prices.append(external_state[PRICE_KEY])
        timestamps.append(external_state[TIMESTAMP_KEY])
        battery_actions.append(charge_kW)
        solar_actions.append(solar_kW_to_battery)
        pv_inputs.append(pv_power)

        external_state, internal_state = battery_environment.step(charge_kW, solar_kW_to_battery, pv_power)

        profits.append(internal_state['total_profit'])
        socs.append(internal_state['battery_soc'])

        if external_state is None:
            break


    rundown_profits = run_down_battery(battery_environment, market_prices)

    return {
        'profits': profits,
        'socs': socs,
        'market_prices': market_prices,
        'actions': battery_actions,
        'solar_actions': solar_actions,
        'pv_inputs': pv_inputs,
        'final_soc': socs[-1],
        'rundown_profit_deltas': rundown_profits,
        'timestamps': timestamps
    }

def parse_parameters(params_list):
    params = {}
    for item in params_list:
        key, value = item.split('=')
        params[key] = eval(value)
    return params

# def perform_eval(args):
#     start = time.time()

#     if args.class_name:
#         policy_config = {'class_name': args.class_name, 'parameters': parse_parameters(args.param)}
#     else:
#         policy_config = load_config("./bot/config.json")

#     policy_class = policy_classes[policy_config['class_name']]
    
#     external_states = pd.read_csv(args.data)
#     if args.output_file:
#         output_file = args.output_file
#     else:
#         results_dir = './results'
#         os.makedirs(results_dir, exist_ok=True)
#         output_file = os.path.join(results_dir, f'{datetime.now().strftime("%Y%m%d_%H%M%S")}_{policy_config["class_name"]}.json')

#     initial_profit = args.initial_profit if 'initial_profit' in args and args.initial_profit is not None else 0
#     initial_soc = args.initial_soc if 'initial_soc' in args and args.initial_profit is not None else 7.5

#     set_seed(args.seed)
#     start_step = args.present_index

#     historical_data = external_states.iloc[:start_step]
#     future_data = external_states.iloc[start_step:]

#     battery_environment = BatteryEnv(
#         data=future_data,
#         initial_charge_kWh=initial_soc,
#         initial_profit=initial_profit
#     )

#     policy = policy_class(**policy_config.get('parameters', {}))
#     policy.load_historical(historical_data)
#     trial_data = run_trial(battery_environment, policy)

#     total_profits = trial_data['profits']
#     rundown_profit_deltas = trial_data['rundown_profit_deltas']

#     mean_profit = float(np.mean(total_profits))
#     std_profit = float(np.std(total_profits))

#     mean_combined_profit = total_profits[-1] + np.sum(rundown_profit_deltas)

#     outcome = {
#         'class_name': policy_config['class_name'],
#         'parameters': policy_config.get('parameters', {}),
#         'mean_profit': mean_profit,
#         'std_profit': std_profit,
#         'score': mean_combined_profit,
#         'main_trial': trial_data,
#         'seconds_elapsed': time.time() - start 
#     }

#     print(f'Average profit ($): {mean_profit:.2f} ± {std_profit:.2f}')
#     print(f'Average profit inc rundown ($): {mean_combined_profit:.2f}')

#     with open(output_file, 'w') as file:
#         json.dump(outcome, file, indent=2)

#     if args.plot:
#         plot_results(trial_data['profits'], trial_data['market_prices'], trial_data['socs'], trial_data['actions'])



# def perform_eval(args):
    start = time.time()

    if args.class_name:
        policy_config = {'class_name': args.class_name, 'parameters': parse_parameters(args.param)}
    else:
        policy_config = load_config("./bot/config.json")

    policy_class = policy_classes[policy_config['class_name']]
    
    external_states = pd.read_csv(args.data)
    if args.output_file:
        output_file = args.output_file
    else:
        results_dir = './results'
        os.makedirs(results_dir, exist_ok=True)
        output_file = os.path.join(results_dir, f'{datetime.now().strftime("%Y%m%d_%H%M%S")}_{policy_config["class_name"]}.json')

    initial_profit = args.initial_profit if 'initial_profit' in args and args.initial_profit is not None else 0
    initial_soc = args.initial_soc if 'initial_soc' in args and args.initial_profit is not None else 7.5

    set_seed(args.seed)
    start_step = args.present_index

    historical_data = external_states.iloc[:start_step]
    future_data = external_states.iloc[start_step:]

    battery_environment = BatteryEnv(
        data=future_data,
        initial_charge_kWh=initial_soc,
        initial_profit=initial_profit
    )
    
    # Reset the environment before the trial starts
    battery_environment.reset()


    policy = policy_class(**policy_config.get('parameters', {}))
    policy.load_historical(historical_data)
    trial_data = run_trial(battery_environment, policy)

    total_profits = trial_data['profits']
    rundown_profit_deltas = trial_data['rundown_profit_deltas']

    mean_profit = float(np.mean(total_profits))
    std_profit = float(np.std(total_profits))

    mean_combined_profit = total_profits[-1] + np.sum(rundown_profit_deltas)

    # Set up the ranges for the grid search
    window_size_norm = 274
    num_std_dev_norm = 0.1
    sec_expo_num_norm = 1
    first_expo_num_norm = 15
    window_size_options_peak_opts = [900]
    num_std_dev_peak = 1.05
    sec_expo_num_peak = 10
    first_expo_num_peak = 15


    # Placeholder for the best parameter combination and its performance
    best_params = None
    best_profit = -np.inf
    best_std_dev = None
    best_trial_data = None

    # Loop over all combinations of hyperparameters for the grid search
    for window_size_peak in window_size_options_peak_opts:
            
        # Reset the environment before the trial starts
        battery_environment.reset()
        
        # Create new policy instance with current grid search parameters
        current_policy_params = {'window_size_norm': window_size_norm,
                                    'num_std_dev_norm': num_std_dev_norm,
                                    'expo_norm': (first_expo_num_norm, sec_expo_num_norm),
                                    
                                    'window_size_peak': window_size_peak,
                                    'num_std_dev_peak': num_std_dev_peak,
                                    'expo_peak': (first_expo_num_peak, sec_expo_num_peak)}
        # Merge with other parameters that might be provided through command line or config
        combined_params = {**policy_config.get('parameters', {}), **current_policy_params}

        policy = policy_class(**combined_params)
        # Load historical data to the policy (if necessary)
        policy.load_historical(historical_data)
        # Run the trial with the current policy
        trial_data = run_trial(battery_environment, policy)
        # # Calculate the total profit for this trial
        # total_profit = trial_data['profits'][-1] + np.sum(trial_data['rundown_profit_deltas'])

        total_profits = trial_data['profits']
        rundown_profit_deltas = trial_data['rundown_profit_deltas']

        mean_profit = float(np.mean(total_profits))
        std_profit = float(np.std(total_profits))

        mean_combined_profit = total_profits[-1] + np.sum(rundown_profit_deltas)


        print(f"Window Size Norm: {window_size_norm}, Num Std Dev Norm: {num_std_dev_norm}, First Expo Number Norm: {first_expo_num_norm}, Second Expo Number Norm: {sec_expo_num_norm}, Window Size Peak: {window_size_peak}, Num Std Dev Peak: {num_std_dev_peak}, First Expo Number Peak: {first_expo_num_peak}, Second Expo Number Peak: {sec_expo_num_peak}, Average profit ($): {mean_profit:.2f} ± {std_profit:.2f}, Average profit inc rundown ($): {mean_combined_profit:.2f}")
        
        # Check if the current combination is better than what we have seen so far
        if mean_profit > best_profit:
            best_profit = mean_profit
            best_params = {'window_size_norm': window_size_norm, 'num_std_dev_norm': num_std_dev_norm, 'expo_norm': (first_expo_num_norm, sec_expo_num_norm), 'window_size_peak': window_size_peak, 'num_std_dev_peak': num_std_dev_peak, 'expo_peak': (first_expo_num_peak, sec_expo_num_peak)}
            best_trial_data = trial_data
            best_std_dev = np.std(trial_data['profits'])

    # Print out the best parameters and the profit achieved with them
    print(f"Best Parameters: {best_params}")
    print(f"Best Profit: {best_profit:.2f}")


    outcome = {
        'class_name': policy_config['class_name'],
        'parameters': policy_config.get('parameters', {}),
        'mean_profit': mean_profit,
        'std_profit': std_profit,
        'score': mean_combined_profit,
        'main_trial': trial_data,
        'seconds_elapsed': time.time() - start 
    }
    

    # print(f'Average profit ($): {mean_profit:.2f} ± {std_profit:.2f}')
    # print(f'Average profit inc rundown ($): {mean_combined_profit:.2f}')

    with open(output_file, 'w') as file:
        json.dump(outcome, file, indent=2)

    if args.plot:
        plot_results(trial_data['profits'], trial_data['market_prices'], trial_data['socs'], trial_data['actions'])

def perform_eval(args):
    start = time.time()

    if args.class_name:
        policy_config = {'class_name': args.class_name, 'parameters': parse_parameters(args.param)}
    else:
        policy_config = load_config("./bot/config.json")

    policy_class = policy_classes[policy_config['class_name']]
    
    external_states = pd.read_csv(args.data)
    if args.output_file:
        output_file = args.output_file
    else:
        results_dir = './results'
        os.makedirs(results_dir, exist_ok=True)
        output_file = os.path.join(results_dir, f'{datetime.now().strftime("%Y%m%d_%H%M%S")}_{policy_config["class_name"]}.json')

    initial_profit = args.initial_profit if 'initial_profit' in args and args.initial_profit is not None else 0
    initial_soc = args.initial_soc if 'initial_soc' in args and args.initial_profit is not None else 7.5

    set_seed(args.seed)
    start_step = args.present_index

    historical_data = external_states.iloc[:start_step]
    future_data = external_states.iloc[start_step:]

    battery_environment = BatteryEnv(
        data=future_data,
        initial_charge_kWh=initial_soc,
        initial_profit=initial_profit
    )
    
    battery_environment.reset()

    # Define the space of hyperparameters to search
    space = {
        'window_size_norm': hp.choice('window_size_norm', range(200, 500, 2)),
        'num_std_dev_norm': hp.uniform('num_std_dev_norm', 0.1, 4.0),
        'sec_expo_num_norm': hp.uniform('sec_expo_num_norm', 1, 20),
        'first_expo_num_norm': hp.uniform('first_expo_num_norm', 10, 60),
        'window_size_peak': hp.choice('window_size_peak', range(200, 1200, 2)),
        'num_std_dev_peak': hp.uniform('num_std_dev_peak', 0.1, 4.0),
        'sec_expo_num_peak': hp.uniform('sec_expo_num_peak', 1, 20),
        'first_expo_num_peak': hp.uniform('first_expo_num_peak', 10, 60)
    }

    # Objective function for hyperopt
    def objective(params):
        battery_environment.reset()
        current_policy_params = {
            'window_size_norm': params['window_size_norm'],
            'num_std_dev_norm': params['num_std_dev_norm'],
            'expo_norm': (params['first_expo_num_norm'], params['sec_expo_num_norm']),
            'window_size_peak': params['window_size_peak'],
            'num_std_dev_peak': params['num_std_dev_peak'],
            'expo_peak': (params['first_expo_num_peak'], params['sec_expo_num_peak'])
        }

        combined_params = {**policy_config.get('parameters', {}), **current_policy_params}

        policy = policy_class(**combined_params)
        policy.load_historical(historical_data)
        trial_data = run_trial(battery_environment, policy)

        total_profits = trial_data['profits']
        rundown_profit_deltas = trial_data['rundown_profit_deltas']
        
        mean_profit = float(np.mean(total_profits))
        std_profit = float(np.std(total_profits))

        mean_combined_profit = total_profits[-1] + np.sum(rundown_profit_deltas)

        
        print(f"Testing combination: {current_policy_params} -> Average profit ($): {mean_profit:.2f} ± {std_profit:.2f}, Average profit inc rundown ($): {mean_combined_profit:.2f}")

        return {'loss': -mean_profit, 'status': STATUS_OK}

    trials = Trials()
    best = fmin(fn=objective,
                space=space,
                algo=tpe.suggest,
                max_evals=50000,
                trials=trials)

    # Extracting the best trial's details
    best_trial_result = trials.best_trial['result']
    best_profit = -best_trial_result['loss']  # Since we minimized the negative profit

    print("Best parameters found: ", best)
    print(f"Best Profit: {best_profit:.2f}")

    outcome = {
        'class_name': policy_config['class_name'],
        'parameters': policy_config.get('parameters', {}),
        'best_parameters': best,
        'best_profit': best_profit,
        'seconds_elapsed': time.time() - start
    }

    with open(output_file, 'w') as file:
        json.dump(outcome, file, indent=2)

    # if args.plot:
    #     # You might need to run a final trial with the best parameters to plot the results
    #     plot_results(trial_data['profits'], trial_data['market_prices'], trial_data['socs'], trial_data['actions'])

    print(f"Execution time: {time.time() - start:.2f} seconds")



def main():
    parser = argparse.ArgumentParser(description='Evaluate a single energy market strategy.')
    parser.add_argument('--plot', action='store_true', help='Plot the results of the main trial.', default=True)
    parser.add_argument('--present_index', type=int, default=0, help='Index to split the historical data from the data which will be used for the evaluation.')
    parser.add_argument('--seed', type=int, default=42, help='Seed for randomness')
    parser.add_argument('--data', type=str, default='./bot/data/validation_data.csv', help='Path to the market data csv file')
    parser.add_argument('--class_name', type=str, help='Policy class name. If not provided, the config.json policy will be used.')
    parser.add_argument('--output_file', type=str, help='File to save all the submission outputs to.', default=None)
    parser.add_argument('--param', action='append', help='Policy parameters as key=value pairs', default=[])
    parser.add_argument('--initial_soc', type=float_or_none, help='Initial state of charge of the battery in kWh', default=None)
    parser.add_argument('--initial_profit', type=float_or_none, help='Initial profit of the battery in $', default=None)

    args = parser.parse_args()

    perform_eval(args)

if __name__ == '__main__':
    main()