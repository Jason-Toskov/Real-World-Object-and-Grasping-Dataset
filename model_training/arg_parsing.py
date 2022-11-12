import argparse

def parse_args():
    parser = argparse.ArgumentParser('')
    
    # Train type
    parser.add_argument('--train_type', type=str, default='cross_val', help='train type')
    
    # General
    parser.add_argument('--config', type=str, default='config.json', help='Path to config file')
    parser.add_argument('--gpu', type=bool, default=True, help='Use GPU')
    parser.add_argument('--output_path', type=str, default='model_training/output/', help='Path to output folder')

    # Dataset
    parser.add_argument('--dataset', type=str, default='grasp_ds', help='Dataset to use')
    parser.add_argument('--num_points', type=int, default=2500, help='Number of points to sample')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--train_frac', type=float, default=0.8, help='Fraction of data to use for training')
    parser.add_argument('--data_augmentation', type=bool, default=False, help='Use data augmentation')
    parser.add_argument('--view_from_grasp', type=bool, default=True, help='Use view from grasp')
    parser.add_argument('--use_normals', type=bool, default=True, help='Use normals')
    parser.add_argument('--cross_val_num', type=int, default=0, help='Cross validation number')
    parser.add_argument('--cross_val_k', type=int, default=5, help='Total number of cross validations')
    parser.add_argument('--separate_stable_unstable', type=bool, default=False, help='Separate stable and unstable grasps')

    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loader')

    # Model
    parser.add_argument('--model', type=str, default='pointnet2', help='Model to use')
    parser.add_argument('--feature_transform', type=bool, default=False, help='Use feature transform')

    # Optimizer
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='Decay rate')

    # LR Scheduler
    parser.add_argument('--step_size', type=int, default=20, help='Step size for LR scheduler')
    parser.add_argument('--gamma', type=float, default=0.7, help='Gamma for LR scheduler')

    # Training
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of epochs to train for')


    return parser.parse_args()