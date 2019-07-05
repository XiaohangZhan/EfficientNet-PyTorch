import argparse
import os
import math
import time
import random
import numpy as np

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torchvision.transforms as transforms
import torchvision.models as models
import torch.backends.cudnn as cudnn
from torch.utils.data.sampler import Sampler
from torch.nn import functional as F

from PIL import Image
from efficientnet_pytorch import EfficientNet

import pdb

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_root', metavar='DIR',
                    help='path to dataset')
    parser.add_argument('data_list', metavar='DIR',
                    help='path to dataset')
    parser.add_argument('--output')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    help='model architecture (default: resnet18)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--action', required=True, type=int)
    parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://10.5.30.223:23456', type=str,
                    help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
    parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
    
    args = parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(extract, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        task(args.gpu, ngpus_per_node, args)

def task(gpu, ngpus_per_node, args):

    args.gpu = gpu
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        #dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
        #                        world_size=args.world_size, rank=args.rank)
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # create model
    if 'efficientnet' in args.arch:
        model = EfficientNet.from_pretrained(args.arch)
    else:
        model = models.__dict__[args.arch](pretrained=True)

    if args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int(args.workers / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    cudnn.benchmark = True

    # data
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    if 'efficientnet' in args.arch:
        image_size = EfficientNet.get_image_size(args.arch)
        dataset = ImageFileList(
            args.data_root,
            args.data_list,
            transforms.Compose([
                transforms.Resize(image_size, interpolation=Image.BICUBIC),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                normalize,
            ]))
    else:
        dataset = ImageFileList(
            args.data_root,
            args.data_list,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]))

    if args.distributed:
        sampler = DistributedSequentialSampler(dataset, args.world_size, args.rank)
    else:
        sampler = None

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, sampler=sampler)

    # extract
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    progress = ProgressMeter(len(loader), batch_time,
                             prefix='Test: ')

    model.eval()
    end = time.time()

    features = []
    for i, images in enumerate(loader):
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)

        with torch.no_grad():
            if args.action == 'extract':
                if args.distributed:
                    feat = F.adaptive_avg_pool2d(
                        model.module.extract_features(images), 1).squeeze(-1).squeeze(-1)
                else:
                    feat = F.adaptive_avg_pool2d(
                        model.extract_features(images), 1).squeeze(-1).squeeze(-1)
            else:
                feat = torch.softmax(model(images), dim=1)
        features.append(feat.cpu().numpy())
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.print(i)

    features = np.concatenate(features, axis=0)
    if args.distributed:
        all_features = gather_tensors_batch(features, 100)
        if args.rank == 0:
            np.save(args.output, all_features)
    else:
        np.save(args.output, features)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def print(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def gather_tensors(input_array):
    world_size = dist.get_world_size()
    ## gather shapes first
    myshape = input_array.shape
    mycount = input_array.size
    shape_tensor = torch.Tensor(np.array(myshape)).cuda()
    all_shape = [torch.Tensor(np.array(myshape)).cuda() for i in range(world_size)]
    dist.all_gather(all_shape, shape_tensor)
    ## compute largest shapes
    all_shape = [x.cpu().numpy() for x in all_shape]
    all_count = [int(x.prod()) for x in all_shape]
    all_shape = [list(map(int, x)) for x in all_shape]
    max_count = max(all_count)
    ## padding tensors and gather them
    output_tensors = [torch.Tensor(max_count).cuda() for i in range(world_size)]
    padded_input_array = np.zeros(max_count)
    padded_input_array[:mycount] = input_array.reshape(-1)
    input_tensor = torch.Tensor(padded_input_array).cuda()
    dist.all_gather(output_tensors, input_tensor)
    ## unpadding gathered tensors
    padded_output = [x.cpu().numpy() for x in output_tensors]
    output = [x[:all_count[i]].reshape(all_shape[i]) for i,x in enumerate(padded_output)]
    return output

def gather_tensors_batch(input_array, part_size=10):
    # gather
    rank = dist.get_rank()
    all_features = []
    part_num = input_array.shape[0] // part_size + 1 if input_array.shape[0] % part_size != 0 else input_array.shape[0] // part_size
    for i in range(part_num):
        part_feat = input_array[i * part_size:min((i+1)*part_size, input_array.shape[0]),...]
        assert part_feat.shape[0] > 0, "rank: {}, length of part features should > 0".format(rank)
        print("rank: {}, gather part: {}/{}, length: {}".format(rank, i, part_num, len(part_feat)))
        gather_part_feat = gather_tensors(part_feat)
        all_features.append(gather_part_feat)
    print("rank: {}, gather done.".format(rank))
    all_features = np.concatenate([np.concatenate([all_features[i][j] for i in range(part_num)], axis=0) for j in range(len(all_features[0]))], axis=0)
    return all_features

class ImageFileList(torch.utils.data.Dataset):

    def __init__(self, root, flist, transform=None, target_transform=None):
        self.root = root
        with open(flist, 'r') as f:
            fns = f.readlines()
            self.imlist = [os.path.join(root, fn.strip()) for fn in fns]
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img = Image.open(self.imlist[index]).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.imlist)


class DistributedSequentialSampler(Sampler):

    def __init__(self, dataset, world_size=None, rank=None):
        if world_size == None:
            world_size = dist.get_world_size()
        if rank == None:
            rank = dist.get_rank()
        self.dataset = dataset
        self.world_size = world_size
        self.rank = rank
        assert len(self.dataset) >= self.world_size, '{} vs {}'.format(len(self.dataset), self.world_size)
        sub_num = int(math.ceil(len(self.dataset) * 1.0 / self.world_size))
        self.beg = sub_num * self.rank
        self.end = self.beg + sub_num
        self.padded_ind = list(range(len(self.dataset))) + list(range(sub_num * self.world_size - len(self.dataset)))

    def __iter__(self):
        indices = [self.padded_ind[i] for i in range(self.beg, self.end)]
        return iter(indices)

    def __len__(self):
        return self.end - self.beg

if __name__ == "__main__":
    main()
