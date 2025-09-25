import torch
import torch.nn as nn
import numpy as np
from typing import List, Type
from dataclasses import dataclass
from collections import defaultdict
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os
from datetime import datetime
from itertools import product
import random

import pycona as ca
from pycona.benchmarks import *
import cpmpy as cp

# Reproducibility
np.random.seed(42)
torch.manual_seed(42)

@dataclass
class ModelConfig:
    hidden_layers: List[int]  # List of neurons in each hidden layer
    activation: Type[nn.Module]
    optimizer_name: str
    learning_rate: float
    name: str
    dropout: float = 0.0
    weight_decay: float = 0.0
    use_weighted_loss: bool = True

def generate_all_configs(
    hidden_layer_sizes=[[64], [64, 32], [64, 32, 16], [128], [128, 64], [128, 64, 32], [32], [32, 16]],
    activations=[nn.ReLU, nn.Tanh, nn.LeakyReLU, nn.Sigmoid],
    optimizers=['adam', 'sgd', 'rmsprop'],
    learning_rates=[0.1, 0.01, 0.001, 0.0001]
):
    """Generate all possible combinations of hyperparameters"""
    configs = []
    
    # Generate all combinations
    combinations = product(
        hidden_layer_sizes,
        activations,
        optimizers,
        learning_rates
    )
    
    # Create configs for each combination
    for hidden_layers, activation, optimizer, lr in combinations:
        # Create descriptive name
        layers_str = '-'.join(str(x) for x in hidden_layers)
        name = f"{layers_str} ({activation.__name__} + {optimizer.upper()} lr={lr})"
        
        config = ModelConfig(
            hidden_layers=hidden_layers,
            activation=activation,
            optimizer_name=optimizer,
            learning_rate=lr,
            name=name
        )
        configs.append(config)
    
    return configs

class TunableNeuralNetClassifier:
    def __init__(self, input_size: int, config: ModelConfig):
        self.input_size = input_size
        self.config = config
        self.output_size = 2  # Binary classification
        self.model = self._build_model()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = self._get_optimizer()
        self.training_history = defaultdict(list)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def _build_model(self) -> nn.Module:
        layers = []
        prev_size = self.input_size
        
        # Add hidden layers
        for hidden_size in self.config.hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(self.config.activation())
            if self.config.dropout and self.config.dropout > 0:
                layers.append(nn.Dropout(self.config.dropout))
            prev_size = hidden_size
        
        # Add output layer (raw logits; softmax applied in predict_proba)
        layers.append(nn.Linear(prev_size, self.output_size))
        
        return nn.Sequential(*layers)
    
    def _get_optimizer(self):
        if self.config.optimizer_name.lower() == 'adam':
            return torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        elif self.config.optimizer_name.lower() == 'sgd':
            return torch.optim.SGD(self.model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        elif self.config.optimizer_name.lower() == 'rmsprop':
            return torch.optim.RMSprop(self.model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer_name}")
    
    def fit(self, X, y, epochs=100):
        self.model.train()

        # Optionally compute class weights for imbalanced data
        if self.config.use_weighted_loss:
            y_np = np.asarray(y)
            # Support binary labels 0..C-1; ensure length equals output_size
            class_counts = np.bincount(y_np, minlength=self.output_size)
            weights = np.zeros(self.output_size, dtype=np.float32)
            nonzero = class_counts > 0
            if nonzero.any():
                weights[nonzero] = class_counts[nonzero].sum() / (self.output_size * class_counts[nonzero])
            # For classes absent in this fold, keep weight 0 to avoid NaNs
            class_weights_tensor = torch.tensor(weights, dtype=torch.float32, device=self.device)
            self.criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        y = torch.tensor(y, dtype=torch.long, device=self.device)
        
        for epoch in range(epochs):
            # Forward pass
            outputs = self.model(X)
            loss = self.criterion(outputs, y)
            
            # Store metrics
            self.training_history['loss'].append(loss.item())
            
            # Backward pass and optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
    
    def predict_proba(self, X):
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X)
            probs = torch.softmax(logits, dim=1)
            return probs.detach().cpu().numpy()
    
    def predict(self, X):
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

def evaluate_model(
    config: ModelConfig,
    X,
    y,
    n_splits: int = 5,
    epochs: int = 100,
    verbose: bool = False,
    reverse_cv: bool = False,
):
    """Evaluate a model configuration using k-fold cross validation.

    If reverse_cv is True, train on a single fold and test on the remaining folds.
    Otherwise, the normal behavior is to train on k-1 folds and test on the held-out fold.
    """
    # Ensure numpy arrays
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)

    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Store metrics for each fold
    fold_metrics = defaultdict(list)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y), 1):
        if verbose:
            print(f"\nFold {fold}/{n_splits}")
        
        # Split data (optionally reverse: train on the small fold, test on the large complement)
        if reverse_cv:
            X_train, X_test = X[val_idx], X[train_idx]
            y_train, y_test = y[val_idx], y[train_idx]
        else:
            X_train, X_test = X[train_idx], X[val_idx]
            y_train, y_test = y[train_idx], y[val_idx]

        # Scale features based on training fold only
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train).astype(np.float32)
        X_test = scaler.transform(X_test).astype(np.float32)
        
        # Create and train classifier
        classifier = TunableNeuralNetClassifier(
            input_size=X.shape[1],
            config=config
        )
        classifier.fit(X_train, y_train, epochs=epochs)
        
        # Get predictions
        y_pred = classifier.predict(X_test)
        
        # Calculate metrics
        fold_metrics['accuracy'].append(accuracy_score(y_test, y_pred))
        fold_metrics['balanced_accuracy'].append(balanced_accuracy_score(y_test, y_pred))
        avg = 'binary' if np.unique(y_test).size == 2 else 'macro'
        fold_metrics['f1'].append(f1_score(y_test, y_pred, average=avg))
        
        if verbose:
            print(f"Fold {fold} metrics:")
            print(f"  Accuracy: {fold_metrics['accuracy'][-1]:.4f}")
            print(f"  Balanced Accuracy: {fold_metrics['balanced_accuracy'][-1]:.4f}")
            print(f"  F1 Score: {fold_metrics['f1'][-1]:.4f}")
    
    # Calculate mean and std of metrics
    results = {
        'config_name': config.name,
        'hidden_layers': str(config.hidden_layers),
        'activation': config.activation.__name__,
        'optimizer': config.optimizer_name,
        'learning_rate': config.learning_rate,
        'accuracy_mean': np.mean(fold_metrics['accuracy']),
        'accuracy_std': np.std(fold_metrics['accuracy']),
        'balanced_accuracy_mean': np.mean(fold_metrics['balanced_accuracy']),
        'balanced_accuracy_std': np.std(fold_metrics['balanced_accuracy']),
        'f1_mean': np.mean(fold_metrics['f1']),
        'f1_std': np.std(fold_metrics['f1']),
        'fold_metrics': dict(fold_metrics)
    }
    
    if verbose:
        print(f"\nOverall results for {config.name}:")
        print(f"Accuracy: {results['accuracy_mean']:.4f} ± {results['accuracy_std']:.4f}")
        print(f"Balanced Accuracy: {results['balanced_accuracy_mean']:.4f} ± {results['balanced_accuracy_std']:.4f}")
        print(f"F1 Score: {results['f1_mean']:.4f} ± {results['f1_std']:.4f}")
    
    return results

def save_results_to_csv(results, experiment_name=None):
    """Save results to CSV files"""
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Generate timestamp for unique filenames
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_name = f"_{experiment_name}" if experiment_name else ""
    
    # Prepare summary data
    summary_data = []
    detailed_data = []
    
    for result in results:
        # Summary row
        summary_row = {
            'config_name': result['config_name'],
            'hidden_layers': result['hidden_layers'],
            'activation': result['activation'],
            'optimizer': result['optimizer'],
            'learning_rate': result['learning_rate'],
            'accuracy_mean': result['accuracy_mean'],
            'accuracy_std': result['accuracy_std'],
            'balanced_accuracy_mean': result['balanced_accuracy_mean'],
            'balanced_accuracy_std': result['balanced_accuracy_std'],
            'f1_mean': result['f1_mean'],
            'f1_std': result['f1_std']
        }
        summary_data.append(summary_row)
        
        # Detailed rows (per fold)
        for fold in range(len(result['fold_metrics']['accuracy'])):
            detailed_row = {
                'config_name': result['config_name'],
                'fold': fold + 1,
                'accuracy': result['fold_metrics']['accuracy'][fold],
                'balanced_accuracy': result['fold_metrics']['balanced_accuracy'][fold],
                'f1': result['fold_metrics']['f1'][fold]
            }
            detailed_data.append(detailed_row)
    
    # Save to CSV files
    summary_df = pd.DataFrame(summary_data)
    detailed_df = pd.DataFrame(detailed_data)
    
    summary_file = f'results/reverse_summary{experiment_name}_{timestamp}.csv'
    detailed_file = f'results/reverse_detailed{experiment_name}_{timestamp}.csv'
    
    summary_df.to_csv(summary_file, index=False)
    detailed_df.to_csv(detailed_file, index=False)
    
    print(f"\nResults saved to:")
    print(f"Summary: {summary_file}")
    print(f"Detailed: {detailed_file}")
    
    return summary_file, detailed_file

def tune_models(X, y, configs, n_splits=5, epochs=100, experiment_name=None, reverse_cv: bool = False):
    """Run tuning experiments with the provided dataset"""
    # Convert inputs to numpy arrays if they aren't already
    X = np.array(X)
    y = np.array(y)
    
    # Evaluate all configurations
    results = []
    for config in configs:
        print(f"\nTesting configuration: {config.name}")
        if reverse_cv:
            print("  Mode: reverse CV (train on 1 fold, test on remaining folds)")
        result = evaluate_model(
            config,
            X,
            y,
            n_splits=n_splits,
            epochs=epochs,
            verbose=True,
            reverse_cv=reverse_cv,
        )
        results.append(result)
    
    # Save results to CSV
    summary_file, detailed_file = save_results_to_csv(results, experiment_name)
    
    # Print summary
    print("\n=== Summary ===")
    print("Best configurations by different metrics:")
    
    # Best by accuracy
    best_acc = max(results, key=lambda x: x['accuracy_mean'])
    print(f"\nBest Accuracy: {best_acc['config_name']}")
    print(f"Accuracy: {best_acc['accuracy_mean']:.4f} ± {best_acc['accuracy_std']:.4f}")
    
    # Best by balanced accuracy
    best_bal_acc = max(results, key=lambda x: x['balanced_accuracy_mean'])
    print(f"\nBest Balanced Accuracy: {best_bal_acc['config_name']}")
    print(f"Balanced Accuracy: {best_bal_acc['balanced_accuracy_mean']:.4f} ± {best_bal_acc['balanced_accuracy_std']:.4f}")
    
    # Best by F1 score
    best_f1 = max(results, key=lambda x: x['f1_mean'])
    print(f"\nBest F1 Score: {best_f1['config_name']}")
    print(f"F1 Score: {best_f1['f1_mean']:.4f} ± {best_f1['f1_std']:.4f}")
    
    return results, summary_file, detailed_file

# Example usage:
if __name__ == "__main__":
    # You can customize the hyperparameter space:
    custom_configs = generate_all_configs(
        hidden_layer_sizes=[
            [64], [64, 32], [64, 32, 16],  # Small architectures
            [128], [128, 64], [128, 64, 32],  # Medium architectures
            [256], [256, 128], [256, 128, 64]  # Large architectures
        ],
        activations=[nn.ReLU, nn.Tanh, nn.LeakyReLU],
        optimizers=['adam', 'sgd'],
        learning_rates=[0.01, 0.001]
    )
    default_configs = generate_all_configs()

    print("Constructing instance...")
    instance, oracle = construct_sudoku(3,3,9)
    print("Constructing bias...")
    instance.construct_bias()
    #random.shuffle(instance.B)
    print("Featurizing constraints...")
    feature_representation = ca.FeaturesRelDim()
    feature_representation.instance = instance
    X = feature_representation.featurize_constraints(instance.bias)
    y = []
    print("Labeling constraints...")
    for c in instance.bias:
        if c in set(oracle.constraints):
            y.append(1)
        else:
            y.append(0)

    # Or use the default space (set reverse_cv=True to train on one fold and test on the rest):
    results, summary_file, detailed_file = tune_models(
        X,
        y,
        default_configs,
        experiment_name="all_combinations",
        reverse_cv=True,
    )