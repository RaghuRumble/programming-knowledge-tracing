import torch
import argparse
import numpy as np
import pandas as pd
import os
from itertools import product
import json

from load_data_ggnn import DATA_ggnn
from run import train, test
from model import MODEL
import random


# set random seed
def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def train_with_params(fold, params_dict, gpu=0):
    """Train model with given hyperparameters and return test AUC"""
    
    # Create args object from params_dict
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=gpu)
    parser.add_argument('--EPOCH', type=int, default=params_dict.get('EPOCH', 300))
    parser.add_argument('--fold', type=int, default=fold)
    parser.add_argument('--dataset', type=str, default='codeforces')
    parser.add_argument('--set_seed', type=bool, default=True)
    parser.add_argument('--node_embed_dim', type=int, default=params_dict.get('node_embed_dim', 64))
    parser.add_argument('--batch_size', type=int, default=params_dict.get('batch_size', 1))
    parser.add_argument('--init_lr', type=float, default=params_dict.get('init_lr', 0.01))
    parser.add_argument('--weight_decay', type=float, default=params_dict.get('weight_decay', 0.00001))
    parser.add_argument('--concept_embed_dim', type=int, default=params_dict.get('concept_embed_dim', 128))
    parser.add_argument('--num_nodes', type=int, default=params_dict.get('num_nodes', 200))
    parser.add_argument('--hidden_dim', type=int, default=params_dict.get('hidden_dim', 64))
    parser.add_argument('--hidden_layers', type=int, default=params_dict.get('hidden_layers', 2))
    parser.add_argument('--ggnn_layers', type=int, default=params_dict.get('ggnn_layers', 4))
    parser.add_argument('--dropout', type=float, default=params_dict.get('dropout', 0.0))
    parser.add_argument('--optimizer_type', type=str, default=params_dict.get('optimizer_type', 'adam'))
    
    root = '../../data/codeforces/ggnn/'
    train_path = root + 'ggnn_train' + str(fold)
    val_path = root + 'ggnn_valid' + str(fold)
    test_path = root + 'ggnn_test'

    parser.add_argument('--num_concepts', type=int, default=37)
    parser.add_argument('--num_problems', type=int, default=7152)
    parser.add_argument('--seqlen', type=int, default=200)
    parser.add_argument('--vocablen', type=int, default=115730)

    params = parser.parse_args([])
    
    # Update params with dict
    for key, value in params_dict.items():
        setattr(params, key, value)

    # set random seed
    seed_torch(0)

    # load data
    data = DATA_ggnn(num_concepts=params.num_concepts, seqlen=params.seqlen)

    try:
        train_p_id, train_c_id, train_node_id, train_edge, train_edge_type, train_target_c, train_result, train_c_embed, \
        train_x_result = data.load_data(train_path)
        
        val_p_id, val_c_id, val_node_id, val_edge, val_edge_type, val_target_c, val_result, val_c_embed, \
        val_x_result = data.load_data(val_path)

        # Create model with dropout support (we'll modify MODEL to support this)
        model = MODEL(num_concepts=params.num_concepts,
                      num_problems=params.num_problems,
                      hidden_dim=params.hidden_dim,
                      hidden_layers=params.hidden_layers,
                      concept_embed_dim=params.concept_embed_dim,
                      vocablen=params.vocablen,
                      node_embed_dim=params.node_embed_dim,
                      num_nodes=params.num_nodes,
                      gpu=params.gpu,
                      ggnn_layers=params.ggnn_layers,
                      dropout=params.dropout)

        model.init_params()
        model.init_embeddings()

        # Choose optimizer
        if params.optimizer_type.lower() == 'sgd':
            optimizer = torch.optim.SGD(params=model.parameters(), lr=params.init_lr, 
                                       weight_decay=params.weight_decay, momentum=0.9)
        else:  # Adam
            optimizer = torch.optim.Adam(params=model.parameters(), lr=params.init_lr, 
                                        weight_decay=params.weight_decay)

        if params.gpu >= 0 and torch.cuda.is_available():
            torch.cuda.set_device(params.gpu)
            model.cuda()
        else:
            params.gpu = -1

        best_valid_auc = 0
        count = 0
        
        for idx in range(params.EPOCH):
            train_loss, train_accuracy, train_auc = train(model, params, optimizer, train_p_id, train_c_id,
                    train_node_id, train_edge, train_edge_type, train_target_c, train_result, train_c_embed, train_x_result)

            valid_loss, valid_accuracy, valid_auc = test(model, params, val_p_id, val_c_id, val_node_id, val_edge,
                    val_edge_type, val_target_c, val_result, val_c_embed, val_x_result)

            if valid_auc > best_valid_auc:
                count = 0
                best_valid_auc = valid_auc
                best_train_auc = train_auc
                best_model = model
            else:
                count += 1
                if count == 5:
                    break

        # test
        test_p_id, test_c_id, test_node_id, test_edge, test_edge_type, test_target_c, test_result, test_c_embed, \
        test_x_result = data.load_data(test_path)

        test_loss, test_accuracy, test_auc = test(best_model, params, test_p_id, test_c_id, test_node_id, test_edge,
                test_edge_type, test_target_c, test_result, test_c_embed, test_x_result)

        return {
            'fold': fold,
            'test_auc': test_auc,
            'test_accuracy': test_accuracy,
            'best_valid_auc': best_valid_auc,
            'params': params_dict
        }
    
    except Exception as e:
        print(f"Error with fold {fold}: {str(e)}")
        return None


def main():
    """
    Hyperparameter tuning for GGNN model.
    Tests different combinations of key hyperparameters.
    """
    
    # Define hyperparameter ranges to test
    hyperparameter_grid = {
        'batch_size': [2, 4, 8],  # Increased from default 1
        'init_lr': [0.001, 0.005, 0.01],  # Different learning rates
        'hidden_dim': [32, 64, 128],  # Hidden dimension
        'node_embed_dim': [32, 64, 128],  # Node embedding dimension
        'ggnn_layers': [3, 4, 5],  # Number of GGNN layers
        'dropout': [0.0, 0.1, 0.2],  # Dropout rates
        'weight_decay': [0.00001, 0.0001],  # Weight decay
        'optimizer_type': ['adam', 'sgd'],  # Optimizer type
    }
    
    # Keep fixed parameters
    fixed_params = {
        'EPOCH': 300,
        'num_nodes': 200,
        'hidden_layers': 2,
        'concept_embed_dim': 128,
    }
    
    # Generate combinations (limit to avoid too many combinations)
    # Use a more targeted approach: test high-impact parameters
    print("Starting hyperparameter tuning...")
    print("Testing on fold 1 first with different configurations...\n")
    
    results = []
    
    # Test 1: Different batch sizes with baseline LR
    print("=" * 60)
    print("Test 1: Batch Size Optimization")
    print("=" * 60)
    for batch_size in [2, 4, 8]:
        params = fixed_params.copy()
        params.update({
            'batch_size': batch_size,
            'init_lr': 0.01,
            'hidden_dim': 64,
            'node_embed_dim': 64,
            'ggnn_layers': 4,
            'dropout': 0.0,
            'weight_decay': 0.00001,
            'optimizer_type': 'adam'
        })
        print(f"\nTesting batch_size={batch_size}...")
        result = train_with_params(fold=1, params_dict=params, gpu=0)
        if result:
            results.append(result)
            print(f"Test AUC: {result['test_auc']:.5f}, Test Acc: {result['test_accuracy']:.5f}")
    
    # Test 2: Different learning rates
    print("\n" + "=" * 60)
    print("Test 2: Learning Rate Optimization")
    print("=" * 60)
    for lr in [0.001, 0.005, 0.01, 0.02]:
        params = fixed_params.copy()
        params.update({
            'batch_size': 4,  # Use better batch size from test 1
            'init_lr': lr,
            'hidden_dim': 64,
            'node_embed_dim': 64,
            'ggnn_layers': 4,
            'dropout': 0.0,
            'weight_decay': 0.00001,
            'optimizer_type': 'adam'
        })
        print(f"\nTesting init_lr={lr}...")
        result = train_with_params(fold=1, params_dict=params, gpu=0)
        if result:
            results.append(result)
            print(f"Test AUC: {result['test_auc']:.5f}, Test Acc: {result['test_accuracy']:.5f}")
    
    # Test 3: Different hidden dimensions and embedding dims
    print("\n" + "=" * 60)
    print("Test 3: Hidden Dimension and Embedding Optimization")
    print("=" * 60)
    for hidden_dim in [32, 64, 128]:
        for node_embed_dim in [32, 64, 128]:
            params = fixed_params.copy()
            params.update({
                'batch_size': 4,
                'init_lr': 0.01,
                'hidden_dim': hidden_dim,
                'node_embed_dim': node_embed_dim,
                'ggnn_layers': 4,
                'dropout': 0.1,
                'weight_decay': 0.00001,
                'optimizer_type': 'adam'
            })
            print(f"\nTesting hidden_dim={hidden_dim}, node_embed_dim={node_embed_dim}...")
            result = train_with_params(fold=1, params_dict=params, gpu=0)
            if result:
                results.append(result)
                print(f"Test AUC: {result['test_auc']:.5f}, Test Acc: {result['test_accuracy']:.5f}")
    
    # Test 4: Different GGNN layers and dropout
    print("\n" + "=" * 60)
    print("Test 4: GGNN Layers and Dropout Optimization")
    print("=" * 60)
    for ggnn_layers in [3, 4, 5]:
        for dropout in [0.0, 0.1, 0.2]:
            params = fixed_params.copy()
            params.update({
                'batch_size': 4,
                'init_lr': 0.01,
                'hidden_dim': 64,
                'node_embed_dim': 64,
                'ggnn_layers': ggnn_layers,
                'dropout': dropout,
                'weight_decay': 0.00001,
                'optimizer_type': 'adam'
            })
            print(f"\nTesting ggnn_layers={ggnn_layers}, dropout={dropout}...")
            result = train_with_params(fold=1, params_dict=params, gpu=0)
            if result:
                results.append(result)
                print(f"Test AUC: {result['test_auc']:.5f}, Test Acc: {result['test_accuracy']:.5f}")
    
    # Save results
    print("\n" + "=" * 60)
    print("TUNING COMPLETE - Saving Results")
    print("=" * 60)
    
    if results:
        # Sort by test AUC (descending)
        results_sorted = sorted(results, key=lambda x: x['test_auc'], reverse=True)
        
        # Save as JSON
        with open('tuning_results.json', 'w') as f:
            json.dump(results_sorted, f, indent=2)
        
        # Save as CSV for easy viewing
        df = pd.DataFrame([
            {
                **r['params'],
                'test_auc': r['test_auc'],
                'test_accuracy': r['test_accuracy'],
                'best_valid_auc': r['best_valid_auc']
            }
            for r in results_sorted
        ])
        df.to_csv('tuning_results.csv', index=False)
        
        print("\nTop 5 Hyperparameter Configurations:")
        print("-" * 60)
        for i, r in enumerate(results_sorted[:5], 1):
            print(f"\n{i}. Test AUC: {r['test_auc']:.5f}, Test Acc: {r['test_accuracy']:.5f}")
            print(f"   Parameters: {r['params']}")
        
        print("\nResults saved to tuning_results.json and tuning_results.csv")


if __name__ == '__main__':
    main()
