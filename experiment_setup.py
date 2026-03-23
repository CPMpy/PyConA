import os
import sys
import argparse
from enum import Enum
from typing import Tuple
from dataclasses import dataclass
import random
import numpy as np
from pycona.find_scope.findscope_obj import split_proba, split_half

RANDOM_SEEDS = [
    42, 123, 7890, 54321, 9876, 
    31415, 27182, 16180, 14142, 22222, 
    55555, 99999, 12345, 67890, 13579, 
    24680, 11111, 33333, 44444, 66666, 
    77777, 88888, 98765, 43210, 10101, 
    20202, 30303, 40404, 50505, 60606
]

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, "pycona"))

import cpmpy as cp
from pycona.benchmarks import (
    construct_sudoku,
    construct_nurse_rostering,
    construct_examtt_simple,
    construct_gtsudoku,
    construct_latin_squares,
    construct_zebra_problem,
    construct_murder_problem,
)
import pycona as ca


BENCHMARKS = [
    'sudoku',
    'exam_timetable',
    'nurse_rostering',
    'nurse_rostering_normal',
    'gtsudoku',
    'latin_square',
    'zebra',
    'murder',
]
ALGORITHMS = ['growacq', 'quacq', 'adagrowacq', 'quacq_solve', 'growacq_solve', 'adagrowacq_solve']

# Constants
SUDOKU_SIZE = 9
SUDOKU_BLOCK_SIZE = 3

SPORTS_SCHEDULING_TEAMS = 6

class Algorithm(Enum):
    GROWACQ = 'growacq'
    QUACQ = 'quacq'
    ADAGROWACQ = 'adagrowacq'
    QUACQ_SOLVE = 'quacq_solve'
    GROWACQ_SOLVE = 'growacq_solve'
    ADAGROWACQ_SOLVE = 'adagrowacq_solve'

@dataclass
class ExperimentConfig:
    benchmark: str
    algorithm: Algorithm
    n_runs: int = 25
    adaptive_grow: int = 3
    verbose: int = 0

    def validate(self):
        if self.adaptive_grow not in [0, 1, 2, 3, 4, 5]:
            raise ValueError("adaptive_grow must be 0, 1, 2, 3, 4, or 5")
        if self.n_runs <= 0:
            raise ValueError("n_runs must be positive")
        if self.benchmark not in BENCHMARKS:
            raise ValueError(f"Invalid benchmark: {self.benchmark}")
        if self.algorithm.value not in ALGORITHMS:
            raise ValueError(f"Invalid algorithm: {self.algorithm}")
        if self.verbose not in [0, 1, 2, 3]:
            raise ValueError("verbose must be 0, 1, or 2")

def construct_benchmark(config: ExperimentConfig) -> Tuple[any, any]:
    """Constructs the benchmark instance and returns (instance, oracle)"""
    
    if config.benchmark == 'sudoku':
        instance, oracle = construct_sudoku(SUDOKU_BLOCK_SIZE, SUDOKU_BLOCK_SIZE, SUDOKU_SIZE)
    elif config.benchmark == 'gtsudoku':
        instance, oracle = construct_gtsudoku()
    elif config.benchmark == 'nurse_rostering':
        instance, oracle = construct_nurse_rostering()
    elif config.benchmark == 'nurse_rostering_normal':
        instance, oracle = construct_nurse_rostering(3, 7, 18, 5)
    elif config.benchmark == 'exam_timetable':
        instance, oracle = construct_examtt_simple(9,6,9,14)
    elif config.benchmark == 'latin_square':
        instance, oracle = construct_latin_squares(10)
    elif config.benchmark == 'zebra':
        instance, oracle = construct_zebra_problem()
    elif config.benchmark == 'murder':
        instance, oracle = construct_murder_problem()
    else:
        raise ValueError(f"Invalid benchmark: {config.benchmark}")

    return instance, oracle

def run_experiment(config: ExperimentConfig) -> None:
    """Run constraint acquisition experiments with specified parameters."""
    config.validate()

    instance, oracle = construct_benchmark(config)
    
    for run in range(config.n_runs):
        seed = RANDOM_SEEDS[run]
        random.seed(seed)
        np.random.seed(seed)
        if config.verbose > 0:
            print(".")

        split_func = split_half
        find_scope = ca.FindScope2(split_func=split_func)
        findc = ca.FindC()
        env = ca.ProbaActiveCAEnv(
            find_scope=find_scope,
            findc=findc,
            feature_representation=ca.FeaturesRelDimBlock()#,
            #seed=seed
        )

        ga = _create_algorithm(config, env)
        
        #try:
        li = ga.learn(instance, oracle, verbose=config.verbose)
        if config.verbose > 0:
            print(ga.env.metrics.statistics)

        _save_results(ga, config)
        #except Exception as e:
        #    print(f"Error in run: {e}")
        #    input("Press Enter to continue...")

def _create_algorithm(config: ExperimentConfig, env) -> any:
    """Creates the appropriate learning algorithm based on configuration"""
    if config.algorithm == Algorithm.QUACQ:
        return ca.QuAcq(env)
    elif config.algorithm == Algorithm.ADAGROWACQ:
        ga_inner = ca.QuAcq(env)
        return ca.AdaGrowAcq(env, inner_algorithm=ga_inner, adaptive_grow=config.adaptive_grow)
    elif config.algorithm == Algorithm.GROWACQ:
        ga_inner = ca.QuAcq(env)
        return ca.GrowAcq(env, inner_algorithm=ga_inner)
    elif config.algorithm == Algorithm.QUACQ_SOLVE:
        return ca.QuAcqSolve(env)
    elif config.algorithm == Algorithm.GROWACQ_SOLVE:
        ga_inner = ca.QuAcqSolve(env)
        return ca.GrowAcq(env, inner_algorithm=ga_inner)
    elif config.algorithm == Algorithm.ADAGROWACQ_SOLVE:
        ga_inner = ca.QuAcqSolve(env)
        return ca.AdaGrowAcq(env, inner_algorithm=ga_inner, adaptive_grow=config.adaptive_grow)
    else:
        raise ValueError(f"Invalid algorithm: {config.algorithm}")

def _save_results(ga, config: ExperimentConfig):
    # Construct filename components
    adaptive_grow_str = f"_ad{config.adaptive_grow}" if config.adaptive_grow > 0 else ""
    filename = f"{config.benchmark}_{config.algorithm.value}{adaptive_grow_str}.csv"

    # Create results directory
    results_dir = os.path.join("results", config.benchmark)
    os.makedirs(results_dir, exist_ok=True)

    # Write results to file in results directory
    ga.env.metrics.write_to_file(os.path.join(results_dir, filename))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run constraint acquisition experiments')
    parser.add_argument('-b', '--benchmark', 
                       choices=BENCHMARKS,
                       required=True,
                       help='The benchmark to run')
    parser.add_argument('-a', '--algorithm', choices=ALGORITHMS,
                        help='The algorithm to use',
                        required=True)
    parser.add_argument('-n', '--n_runs', type=int, default=25,
                        help='Number of runs to perform (default: 25)')
    parser.add_argument('-ad', '--adaptive-grow', type=int, default=3,
                        help='Adaptive growing strategy to use in growacq (0=disabled, 1=strategy1, 2=strategy2, 3=strategy3, 4=strategy4, 5=strategy5) (default: 3)')
    parser.add_argument('-v', '--verbose', type=int, default=0,
                        help='Verbose level (default: 0)')
    args = parser.parse_args()

    config = ExperimentConfig(
        benchmark=args.benchmark,
        algorithm=Algorithm(args.algorithm),
        n_runs=args.n_runs,
        adaptive_grow=args.adaptive_grow,
        verbose=args.verbose,
    )
    
    run_experiment(config)


