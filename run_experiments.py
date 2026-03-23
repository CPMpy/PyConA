import json
import itertools
from experiment_setup import run_experiment
import time
import logging
from datetime import datetime
import argparse
import os
from experiment_setup import ExperimentConfig, Algorithm

# Centralize experiment logs in project-level logs/
logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(logs_dir, exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(
            os.path.join(logs_dir, f'experiment_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        ),
        logging.StreamHandler()
    ]
)

def load_config(config_file='config.json'):
    """Load experiment configurations from JSON file."""
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Configuration file '{config_file}' not found. Please provide a valid config file path using -c or --config argument.")
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config

def get_parameter_combinations(config):
    """Generate all possible combinations of parameters."""
    # Get all keys except benchmarks and algorithms
    param_keys = [k for k in config.keys() if k not in ['benchmark', 'algorithm']]
    param_values = [config[k] for k in param_keys]
    combinations = list(itertools.product(*param_values))
    return [dict(zip(param_keys, combo)) for combo in combinations]

def run_all_experiments(config_file='config.json'):
    """Run all experiments defined in the config file."""
    config = load_config(config_file)
    
    # Generate all parameter combinations
    parameter_combinations = get_parameter_combinations(config)
    
    # Calculate total number of experiments
    total_experiments = (len(config['benchmark']) * 
                        len(config['algorithm']) * 
                        len(parameter_combinations))
    
    current_experiment = 0
    start_time = time.time()
    
    for benchmark in config['benchmark']:
        for algorithm in config['algorithm']:
            for params in parameter_combinations:
                current_experiment += 1
                
                # Log experiment start
                logging.info(f"\nStarting experiment {current_experiment}/{total_experiments}")
                logging.info(f"Benchmark: {benchmark}")
                logging.info(f"Algorithm: {algorithm}")
                logging.info(f"Parameters: {params}")
                
                try:
                    # Create ExperimentConfig object
                    exp_config = ExperimentConfig(
                        benchmark=benchmark,
                        algorithm=Algorithm(algorithm),
                        **params
                    )
                    
                    # Run the experiment with ExperimentConfig
                    run_experiment(exp_config)
                    logging.info(f"Experiment completed successfully")
                    
                except Exception as e:
                    logging.error(f"Error in experiment: {str(e)}")
                    continue
                
                # Log progress
                elapsed_time = time.time() - start_time
                avg_time_per_exp = elapsed_time / current_experiment
                remaining_experiments = total_experiments - current_experiment
                estimated_remaining_time = avg_time_per_exp * remaining_experiments
                
                logging.info(f"Progress: {current_experiment}/{total_experiments}")
                logging.info(f"Elapsed time: {elapsed_time/3600:.2f} hours")
                logging.info(f"Estimated remaining time: {estimated_remaining_time/3600:.2f} hours")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run experiments using a configuration file')
    parser.add_argument('-c', '--config', type=str, default='config.json',
                       help='Path to the configuration JSON file (default: config.json)')
    args = parser.parse_args()
    
    run_all_experiments(args.config) 