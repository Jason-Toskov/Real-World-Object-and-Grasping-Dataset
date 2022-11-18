import json

from trainer import run_training
from arg_parsing import parse_args

def run_cross_val(args, cfg, base_output_path):
    for i in range(1, args.cross_val_k+1):
        args.cross_val_num = i
        args.output_path = base_output_path + 'cross_val_'+str(i)+'/'
        metric_log = run_training(args, cfg)
        print('Cross val',i,'metric log:',metric_log)

def compare_pose_in_and_transform(args, cfg, base_output_path):
    args.view_from_grasp = False
    args.output_path = base_output_path + 'feed_pose/'
    metric_log = run_training(args, cfg)

    args.view_from_grasp = True
    args.output_path = base_output_path + 'grasp_view/'
    metric_log = run_training(args, cfg)

def test_success_vs_stable_success(args, cfg, base_output_path):
    args.unstable_is_success = False
    args.output_path = base_output_path + 'exclude_unstable/'
    run_cross_val(args, cfg, args.output_path)

    args.unstable_is_success = True
    args.output_path = base_output_path + 'include_unstable/'
    run_cross_val(args, cfg, args.output_path)

if __name__ == '__main__':
    args = parse_args()
    # load config
    with open(args.config, 'r') as f:
        cfg = json.load(f)
    
    if args.train_type == 'default':
        run_training(args, cfg)
    elif args.train_type == 'cross_val':
        run_cross_val(args, cfg, args.output_path)
    elif args.train_type == 'compare_pose_in_and_transform':
        compare_pose_in_and_transform(args, cfg, args.output_path)
    elif args.train_type == 'test_success_vs_stable_success':
        test_success_vs_stable_success(args, cfg, args.output_path)
    else:
        raise ValueError("train_type must be 'cross_val', 'compare_pose_in_and_transform', or 'test_success_vs_stable_success'")
