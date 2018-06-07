import find_mxnet
import mxnet as mx
import argparse
import os, sys
import train_model
import socket #2016.10.6
import linecache

parser = argparse.ArgumentParser(description='train an image classifer on cifar100')
parser.add_argument('--network', type=str, default='inception-bn-28-small',
                    help = 'the cnn to use')
parser.add_argument('--data-dir', type=str, default='cifar100/',
                    help='the input data directory')
parser.add_argument('--gpus', type=str,
                    help='the gpus will be used, e.g "0,1,2,3"')
parser.add_argument('--num-examples', type=int, default=60000,
                    help='the number of training examples')
parser.add_argument('--batch-size', type=int, default=128,
                    help='the batch size')
parser.add_argument('--lr', type=float, default=.05,
                    help='the initial learning rate')
parser.add_argument('--lr-factor', type=float, default=1,
                    help='times the lr with a factor for every lr-factor-epoch epoch')
parser.add_argument('--lr-factor-epoch', type=float, default=1,
                    help='the number of epoch to factor the lr, could be .5')
parser.add_argument('--num-epochs', type=int, default=20,
                    help='the number of training epochs')
parser.add_argument('--kv-store', type=str, default='local',
                    help='the kvstore type')
parser.add_argument('--log-file', type=str,
                    help='the name of log file')
parser.add_argument('--log-dir', type=str, default="output",
                    help='directory of the log file')
'''yegeyan 2016.10.6'''
parser.add_argument('--hostname', type=str, default="gpu-node-1",
                    help='the hostname of this worker')
parser.add_argument('--dataset', type=str, default='cifar100',
                    help='the dataset of training')
parser.add_argument('--staleness', type=int, default=0,
                    help='the staleness of dist_ssync')
parser.add_argument('--savemodel', action='store_true', default=False,
                    help='true means save model')
parser.add_argument('--retrain', action='store_true', default=False,
                    help='true means continue training')
parser.add_argument('--model-load-epoch', type=int, default=0,
                    help='load the model on an epoch using the model-load-prefix')
parser.add_argument('--data-allocator', type=int, default=0,
                    help='whether to use data allocator by group')
args = parser.parse_args()

args.hostname = socket.gethostbyname_ex(socket.gethostname())[0] #yegeyan 2016.10.6

# download data if necessary
def _download(data_dir):
    if not os.path.isdir(data_dir):
        os.system("mkdir " + data_dir)
    os.chdir(data_dir)
    if (not os.path.exists('train.rec')) or \
       (not os.path.exists('test.rec')) :
        os.system("wget http://data.dmlc.ml/mxnet/data/cifar100.zip")
        os.system("unzip -u cifar100.zip")
        os.system("mv cifar/* .; rm -rf cifar; rm cifar100.zip")
    os.chdir("..")

# network
import importlib
net = importlib.import_module('symbol_' + args.network).get_symbol(100)

# data
def get_iterator(args, kv):
    data_shape = (3, 28, 28)
    if '://' not in args.data_dir:
        _download(args.data_dir)
    
    splits = kv.num_workers
    part = kv.rank
    val_splits = kv.num_workers
    val_part = kv.rank

    if args.data_allocator == 1:
        data_rate = linecache.getline("data_sharding", kv.rank + 1).split(' ')
        part = float(data_rate[0])
        splits = float(data_rate[1])
    args.data_proportion = splits - part
        
    train = mx.io.ImageRecordIter(
        path_imgrec = args.data_dir + "train.rec",
        mean_img    = args.data_dir + "mean.bin",
        data_shape  = data_shape,
        batch_size  = args.batch_size,
        rand_crop   = True,
        rand_mirror = True,
        preprocess_threads  = 4,
        num_parts   = splits, #yegeyan 2016.10.6
        part_index  = part) #yegeyan 2016.10.6

    val = mx.io.ImageRecordIter(
        path_imgrec = args.data_dir + "test.rec",
        mean_img    = args.data_dir + "mean.bin",
        rand_crop   = False,
        rand_mirror = False,
        data_shape  = data_shape,
        batch_size  = args.batch_size,
        preprocess_threads  = 4,
        num_parts   = val_splits, #yegeyan 2016.10.6
        part_index  = val_part) #yegeyan 2016.10.6

    return (train, val)

# train
train_model.fit(args, net, get_iterator)
