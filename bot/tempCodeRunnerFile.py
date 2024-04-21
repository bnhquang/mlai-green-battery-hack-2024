                    # Reset the environment before the trial starts
                    battery_environment.reset()

                    
                    # Create new policy instance with current grid search parameters
                    current_policy_params = {'window_size': window_size, 'num_std_dev': num_std_dev, 'expo': (first_expo_num, sec_expo_num)}
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


                    print(f"Window Size: {window_size}, Num Std Dev: {num_std_dev}, Second Expo Number: {sec_expo_num}, Average profit ($): {mean_profit:.2f} ± {std_profit:.2f}, Average profit inc rundown ($): {mean_combined_profit:.2f}")
                    
                    # Check if the current combination is better than what we have seen so far
                    if mean_profit > best_profit:
                        best_profit = mean_profit
                        best_params = {'window_size': window_size, 'num_std_dev': num_std_dev}
                        best_trial_data = trial_data
                        best_std_dev = np.std(trial_data['profits'])
