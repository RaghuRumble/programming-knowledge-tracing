"""
Configuration executor for GGNN model.
Allows running main.py with any configuration from hyperparameter_configs.json
"""

import json
import subprocess
import sys
import os


def load_configs():
    """Load configurations from JSON"""
    with open('hyperparameter_configs.json', 'r') as f:
        return json.load(f)


def run_config(config_name, folds=[1, 2, 3, 4, 5]):
    """
    Run main.py with specific configuration on all folds
    
    Args:
        config_name: Name of configuration from hyperparameter_configs.json
        folds: List of folds to run
    """
    configs = load_configs()
    
    if config_name not in configs:
        print(f"Configuration '{config_name}' not found!")
        print(f"Available configurations: {list(configs.keys())}")
        return
    
    config = configs[config_name]
    
    print(f"\n{'='*70}")
    print(f"Running configuration: {config_name}")
    print(f"{'='*70}")
    print(f"Description: {config.get('description', 'N/A')}")
    print(f"Expected improvement: {config.get('expected_improvement', 'N/A')}")
    print("\nParameters:")
    for key, value in config.items():
        if key not in ['description', 'expected_improvement']:
            print(f"  {key}: {value}")
    
    # Build command for each fold
    results = []
    for fold in folds:
        print(f"\n{'='*50}")
        print(f"Fold {fold}/{len(folds)}")
        print(f"{'='*50}")
        
        # Build command
        cmd = ['python', 'main.py']
        
        # Add configuration parameters
        for key, value in config.items():
            if key not in ['description', 'expected_improvement']:
                if isinstance(value, str):
                    cmd.extend([f'--{key}', value])
                else:
                    cmd.extend([f'--{key}', str(value)])
        
        # Add fold
        cmd.extend(['--fold', str(fold)])
        
        try:
            result = subprocess.run(cmd, capture_output=False)
            if result.returncode != 0:
                print(f"Warning: Fold {fold} exited with code {result.returncode}")
            else:
                results.append(fold)
        except Exception as e:
            print(f"Error running fold {fold}: {str(e)}")
    
    print(f"\n{'='*70}")
    print(f"Configuration '{config_name}' completed")
    print(f"Successfully ran folds: {results}")
    print(f"{'='*70}")


def list_configs():
    """List available configurations"""
    configs = load_configs()
    print("\nAvailable Configurations:")
    print("-" * 70)
    for name, config in configs.items():
        print(f"\n{name}:")
        print(f"  Description: {config.get('description', 'N/A')}")
        print(f"  Batch size: {config.get('batch_size', 'N/A')}")
        print(f"  Learning rate: {config.get('init_lr', 'N/A')}")
        print(f"  Dropout: {config.get('dropout', 'N/A')}")
        print(f"  Hidden dim: {config.get('hidden_dim', 'N/A')}")
        print(f"  GGNN layers: {config.get('ggnn_layers', 'N/A')}")


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("\nUsage:")
        print("  python run_config.py list                 - List available configurations")
        print("  python run_config.py <config_name>        - Run configuration on folds 1-5")
        print("  python run_config.py <config_name> <folds> - Run configuration on specific folds")
        print("\nExample:")
        print("  python run_config.py recommended_v1")
        print("  python run_config.py recommended_v2 1 2 3")
        return
    
    if sys.argv[1] == 'list':
        list_configs()
        return
    
    config_name = sys.argv[1]
    folds = [int(f) for f in sys.argv[2:]] if len(sys.argv) > 2 else [1, 2, 3, 4, 5]
    
    run_config(config_name, folds)


if __name__ == '__main__':
    main()
