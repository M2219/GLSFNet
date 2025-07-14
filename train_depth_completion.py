from __future__ import print_function, division

import os
import time
import gc

import argparse
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import numpy as np

from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from datasets import StereoDataset
from models import __models__, model_loss_train, model_loss_test
from utils import *

cudnn.benchmark = True
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
print(torch.cuda.get_device_name(0))

parser = argparse.ArgumentParser(description='GLSFNet')
parser.add_argument('--model', default='GLSFNet', help='select a model structure', choices=__models__.keys())
parser.add_argument('--backbone', default='efficientnet_b2a', help='select a model structure', choices=["mobilenetv2_100", "efficientnet_b2a"])
parser.add_argument('--maxdisp', type=int, default=192, help='maximum disparity')

parser.add_argument('--dataset', default='kitti_depth_completion', help='dataset name')
parser.add_argument('--datapath_im', default="/datasets/kittiraw/2011_09_26/", help='data path')
parser.add_argument('--datapath_s', default="/datasets/data_depth_velodyne/", help='data path')
parser.add_argument('--datapath_gt', default="/datasets/data_depth_annotated/", help='data path')

parser.add_argument('--trainlist', default='./filenames/kitti_depth_train.txt', help='training list')
parser.add_argument('--testlist',default='./filenames/kitti_depth_val.txt', help='testing list')

parser.add_argument('--lr', type=float, default=0.001, help='base learning rate')
parser.add_argument('--batch_size', type=int, default=4, help='training batch size')
parser.add_argument('--test_batch_size', type=int, default=4, help='testing batch size')
parser.add_argument('--epochs', type=int, default=60, help='number of epochs to train')
parser.add_argument('--lrepochs', type=str, default="20,32,40,48,56:2", help='the epochs to decay lr: the downscale rate')

parser.add_argument('--logdir', default='', help='the directory to save logs and checkpoints')
parser.add_argument('--loadckpt', default='', help='load the weights from a specific checkpoint')
parser.add_argument('--resume', action='store_true', help='continue training the model')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

parser.add_argument('--summary_freq', type=int, default=1, help='the frequency of saving summary')
parser.add_argument('--save_freq', type=int, default=1, help='the frequency of saving checkpoint')
parser.add_argument('--cv_scale', type=int, default=4, help='cost volume scale factor', choices=[8, 4])
parser.add_argument('--cv', type=str, default='gwc', choices=[
          'norm_correlation',
          'gwc',
], help='selecting a cost volumes')

args = parser.parse_args()

gwc = False
norm_correlation = False
if args.cv == 'norm_correlation':
    norm_correlation = True
elif args.cv == 'gwc':
    gwc = True

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
os.makedirs(args.logdir, exist_ok=True)

print("creating new summary file")
logger = SummaryWriter(args.logdir)


train_dataset = StereoDataset(datapath_im=args.datapath_im, datapath_s=args.datapath_s, datapath_gt=args.datapath_gt,
                              list_filename=args.trainlist, kitti_completion=True, is_vkitti2=False, train_status=True)

test_dataset = StereoDataset(datapath_im=args.datapath_im, datapath_s=args.datapath_s, datapath_gt=args.datapath_gt,
                              list_filename=args.testlist, kitti_completion=True, is_vkitti2=False, train_status=False)

TrainImgLoader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=8, drop_last=True)
TestImgLoader = DataLoader(test_dataset, args.test_batch_size, shuffle=False, num_workers=8, drop_last=False)

model = __models__[args.model](args.maxdisp, gwc, norm_correlation, args.backbone, args.cv_scale)
model = nn.DataParallel(model)
model.cuda()

print("The number of parameters:", count_parameters(model))

optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999))

start_epoch = 0
if args.resume:
    all_saved_ckpts = [fn for fn in os.listdir(args.logdir) if fn.endswith(".ckpt")]
    all_saved_ckpts = sorted(all_saved_ckpts, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    loadckpt = os.path.join(args.logdir, all_saved_ckpts[-1])
    print("loading the lastest model in logdir: {}".format(loadckpt))
    state_dict = torch.load(loadckpt)
    model.load_state_dict(state_dict['model'])
    optimizer.load_state_dict(state_dict['optimizer'])
    start_epoch = state_dict['epoch'] + 1
elif args.loadckpt:
    print("loading model {}".format(args.loadckpt))
    state_dict = torch.load(args.loadckpt)
    model_dict = model.state_dict()
    pre_dict = {k: v for k, v in state_dict['model'].items() if k in model_dict}
    model_dict.update(pre_dict)
    model.load_state_dict(model_dict)

print("start at epoch {}".format(start_epoch))

def train():
    bestepoch = 0
    error = 100

    loss_ave = AverageMeter()
    RMSE_ave = AverageMeter()
    EPE_ave = AverageMeter()

    loss_ave_t = AverageMeter()
    RMSE_ave_t = AverageMeter()
    EPE_ave_t = AverageMeter()
    MAE_ave_t = AverageMeter()
    iMAE_ave_t = AverageMeter()
    iRMSE_ave_t = AverageMeter()

    evaluate_performance = False
    if  evaluate_performance:
        dummy_input1 = torch.randn(1, 3, 512, 960, dtype=torch.float).cuda()
        dummy_input2 = torch.randn(1, 3, 512, 960, dtype=torch.float).cuda()
        inference_time = measure_performance(dummy_input1, dummy_input2)
        print("inference time = ", inference_time)
        return 0

    for epoch_idx in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch_idx, args.lr, args.lrepochs)

        for batch_idx, sample in enumerate(TrainImgLoader):

            global_step = len(TrainImgLoader) * epoch_idx + batch_idx
            start_time = time.time()
            do_summary = global_step % args.summary_freq == 0

            loss, scalar_outputs = train_sample(sample, compute_metrics=do_summary)

            loss_ave.update(loss)
            RMSE_ave.update(scalar_outputs['RMSE'][0])
            EPE_ave.update(scalar_outputs['EPE'][0])

            if do_summary:
                save_scalars(logger, 'train', scalar_outputs, global_step)

            print('Epoch {}/{} | Iter {}/{} | train loss = {:.3f}({:.3f}) | RMSE = {:.3f}({:.3f}) | EPE = {:.3f}({:.3f})| time = {:.3f}'.format(epoch_idx, args.epochs,
                                                                                       batch_idx,
                                                                                       len(TrainImgLoader), loss, loss_ave.avg,
                                                                                       scalar_outputs['RMSE'][0], RMSE_ave.avg, scalar_outputs['EPE'][0], EPE_ave.avg,
                                                                                       time.time() - start_time))
            del scalar_outputs
        if (epoch_idx + 1) % args.save_freq == 0:
            checkpoint_data = {'epoch': epoch_idx, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
            torch.save(checkpoint_data, "{}/checkpoint_{:0>6}.ckpt".format(args.logdir, epoch_idx))
        gc.collect()

        avg_test_scalars = AverageMeterDict()
        for batch_idx, sample in enumerate(TestImgLoader):
            global_step = len(TestImgLoader) * epoch_idx + batch_idx
            do_summary = global_step % args.summary_freq == 0

            start_time = time.time()
            loss, scalar_outputs = test_sample(sample, compute_metrics=do_summary)
            tt = time.time()


            loss_ave_t.update(loss)
            RMSE_ave_t.update(scalar_outputs['RMSE'][0])
            EPE_ave_t.update(scalar_outputs['EPE'][0])
            iMAE_ave_t.update(scalar_outputs["iMAE"][0])
            iRMSE_ave_t.update(scalar_outputs["iRMSE"][0])
            MAE_ave_t.update(scalar_outputs["MAE"][0])


            if do_summary:
                save_scalars(logger, 'test', scalar_outputs, global_step)
            avg_test_scalars.update(scalar_outputs)

            print('Epoch {}/{} | Iter {}/{} | val loss = {:.3f}({:.3f}) | RMSE = {:.3f}({:.3f}) | EPE = {:.3f}({:.3f}) | iMAE = {:.3f}({:.3f}) | iRMSE = {:.3f}({:.3f}) | MAE = {:.3f}({:.3f})| time = {:.3f}'.format(epoch_idx, args.epochs,
                                                                                       batch_idx,
                                                                                       len(TestImgLoader), loss, loss_ave_t.avg,
                                                                                       scalar_outputs['RMSE'][0], RMSE_ave_t.avg, scalar_outputs['EPE'][0], EPE_ave_t.avg,
                                                                                       scalar_outputs['iMAE'][0], iMAE_ave_t.avg, scalar_outputs['iRMSE'][0], iRMSE_ave_t.avg,
                                                                                       scalar_outputs['MAE'][0], MAE_ave_t.avg,
                                                                                       tt - start_time))
            del scalar_outputs

        avg_test_scalars = avg_test_scalars.mean()
        nowerror = avg_test_scalars["RMSE"][0]
        if  nowerror < error :
            bestepoch = epoch_idx
            error = avg_test_scalars["RMSE"][0]
        save_scalars(logger, 'fulltest', avg_test_scalars, len(TrainImgLoader) * (epoch_idx + 1))
        print("avg_test_scalars", avg_test_scalars)
        print('MAX epoch %d total test error = %.5f' % (bestepoch, error))
        gc.collect()
    print('MAX epoch %d total test error = %.5f' % (bestepoch, error))

def train_sample(sample, compute_metrics=False):
    model.train()

    imgL, imgR, sparseL, disp_gt, disp_gt_low, depth_gt, conversion_rate, valid = sample['left'], sample['right'], sample['disparity'], \
                                                                sample['disparity_gt'], sample['disparity_gt_low'], sample['depth_gt'], \
                                                                sample["conversion_rate"], sample['valid']

    imgL = imgL.cuda()
    imgR = imgR.cuda()
    sparseL = sparseL.cuda()

    disp_gt = disp_gt.cuda()
    disp_gt_low = [d.cuda() for d in disp_gt_low]
    depth_gt = depth_gt.cuda()
    valid = valid.cuda()


    optimizer.zero_grad()

    depth_est, disp_est, disp_low_est = model(imgL, imgR, sparseL, conversion_rate, train_status=True)
    disp_ests = [disp_est, disp_low_est]

    mask = (disp_gt < args.maxdisp) & (disp_gt > 0)
    mask_low = [(d < args.maxdisp) & (d > 0) for d in disp_gt_low]
    masks = [mask] + mask_low
    disp_gts = [disp_gt] + disp_gt_low

    mask_d = (disp_gt < args.maxdisp) & (valid > 0)

    loss = model_loss_train(depth_est, depth_gt, mask_d, disp_ests, disp_gts, masks, args.cv_scale)
    disp_ests_final = disp_ests[0]

    scalar_outputs = {"loss": loss}
    if compute_metrics:
        with torch.no_grad():
            scalar_outputs["RMSE"] = [RMSE_metric(depth_est, depth_gt, mask_d)]
            scalar_outputs["EPE"] = [EPE_metric(disp_ests_final, disp_gt, mask)]

    loss.backward()
    optimizer.step()

    return tensor2float(loss), tensor2float(scalar_outputs)

@make_nograd_func
def test_sample(sample, compute_metrics=True):
    model.eval()

    imgL, imgR, sparseL, disp_gt, depth_gt, conversion_rate, valid = sample['left'], sample['right'], sample['disparity'], \
                                                    sample['disparity_gt'], sample['depth_gt'], sample["conversion_rate"], sample['valid']

    imgL = imgL.cuda()
    imgR = imgR.cuda()
    sparseL = sparseL.cuda()

    disp_gt = disp_gt.cuda()
    depth_gt = depth_gt.cuda()
    valid = valid.cuda()

    depth_est, disp_est = model(imgL, imgR, sparseL, conversion_rate, train_status=False)

    mask = (disp_gt < args.maxdisp) & (disp_gt > 0)
    masks = [mask]
    disp_gts = [disp_gt]
    mask_d = (disp_gt < args.maxdisp) & (valid > 0)

    loss = model_loss_test(depth_est, depth_gt, mask_d)

    scalar_outputs = {"loss": loss}

    scalar_outputs["iMAE"] = [iMAE_metric(depth_est, depth_gt, mask_d)]
    scalar_outputs["iRMSE"] = [iRMSE_metric(depth_est, depth_gt, mask_d)]
    scalar_outputs["MAE"] = [MAE_metric(depth_est, depth_gt, mask_d)]
    scalar_outputs["RMSE"] = [RMSE_metric(depth_est, depth_gt, mask_d)]
    scalar_outputs["EPE"] = [EPE_metric(disp_est, disp_gt, mask)]

    return tensor2float(loss), tensor2float(scalar_outputs)

@make_nograd_func
def measure_performance(dummy_input1, dummy_input2):
    model.eval()

    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 500
    timings=np.zeros((repetitions,1))
    for _ in range(10):
        _ = model(dummy_input1, dummy_input2, train_status=True)

    for rep in range(repetitions):
        starter.record()
        _ = model(dummy_input1, dummy_input2, train_status=True)
        ender.record()
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)
        timings[rep] = curr_time

    mean_syn = np.sum(timings) / repetitions
    np.std(timings)

    return  mean_syn

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == '__main__':
    train()
