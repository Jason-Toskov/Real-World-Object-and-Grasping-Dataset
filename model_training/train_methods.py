import json

from trainer import run_training
from arg_parsing import parse_args

def run_cross_val(args, cfg):
    for i in range(1, args.cross_val_k+1):
        args.cross_val_num = i
        args.output_path = 'model_training/output/cross_val_'+str(i)+'/'
        metric_log = run_training(args, cfg)
        print('Cross val',i,'metric log:',metric_log)

if __name__ == '__main__':
    args = parse_args()
    # load config
    with open(args.config, 'r') as f:
        cfg = json.load(f)
    
    if args.train_type == 'cross_val':
        run_cross_val(args, cfg)
