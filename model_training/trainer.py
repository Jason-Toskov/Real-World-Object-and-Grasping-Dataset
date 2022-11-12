import json
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from arg_parsing import parse_args
from GraspLoader import GraspingDataset
from models.model import PointNetGraspCls
from models.pointnet2_cls_ssg_grasp_pose import PointNet2Model
from logging_utils import MetricLogger

def fwd_pass(args, cfg, model, pcl, pose, target, device):
    if args.model == 'pointnet2':
        pcl = pcl.transpose(2, 1)
        output, l3_points = model(pcl, pose)
    return output

def train(args, cfg, model, loader, loss, optimizer, device):
    model.train()
    total_loss = 0
    for batch_idx, (pcl, pose, target) in enumerate(loader):
        pcl, pose, target = pcl.to(device), pose.to(device), target.to(device)
        output = fwd_pass(args, cfg, model, pcl, pose, target, device)
        l = loss(output, target)
        l.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += l.item()
    return total_loss/(batch_idx+1)

# TODO: per object evaluation
def test(args, cfg, model, loader, loss, device):
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (pcl, pose, target) in enumerate(loader):
            pcl, pose, target = pcl.to(device), pose.to(device), target.to(device)
            output = fwd_pass(args, cfg, model, pcl, pose, target, device)
            l = loss(output, target)
            total_loss += l.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    return total_loss/(batch_idx+1), 100*correct/len(loader.dataset)

def run_training(args, cfg):    
    OUTPUT_PATH = cfg['dirs']['file_dir']+args.output_path
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)
    
    # save args
    with open(OUTPUT_PATH+'args.json', 'w') as f:
        json.dump(vars(args), f)
    
    # save config
    with open(OUTPUT_PATH+'config.json', 'w') as f:
        json.dump(cfg, f)
    
    # print args nicely
    print('Args:')
    for k, v in vars(args).items():
        print('  ', k, ':', v)

    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() and args.gpu else 'cpu')
    print('Using device:', device)

    # Init dataset
    ds_train = GraspingDataset(args, cfg, split='train')
    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    ds_test = GraspingDataset(args, cfg, split='test')
    dl_test = DataLoader(ds_test, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # Init model
    # TODO: implement feature transform
    # TODO: implement other models
    model = PointNet2Model(len(ds_train.classes), args.use_normals, args.view_from_grasp).to(device)

    # Init optimizer + loss
    optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    loss = torch.nn.NLLLoss()

    metric_log = MetricLogger()

    # Train
    best_test_acc = 0
    pbar = tqdm(range(args.num_epochs))
    for epoch in pbar:
        train_loss = train(args, cfg, model, dl_train, loss, optimizer, device)
        _, train_acc = test(args, cfg, model, dl_train, loss, device)
        test_loss, test_acc = test(args, cfg, model, dl_test, loss, device)
        scheduler.step()

        metric_log.update(train_loss, test_loss, train_acc, test_acc)

        # update tqdm description
        pbar.set_description(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}')

        # save model
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), OUTPUT_PATH+'best_model.py')
    
    # save metrics
    metric_log.save(OUTPUT_PATH)
    metric_log.plot(OUTPUT_PATH)

    return metric_log

if __name__ == '__main__':
    args = parse_args()
    # load config
    with open(args.config, 'r') as f:
        cfg = json.load(f)

    metric_log = run_training(args, cfg)