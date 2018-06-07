import find_mxnet
import argparse,logging,os
import mxnet as mx
from symbol_resnet import resnet
import socket
import getpass

def multi_factor_scheduler(begin_epoch, epoch_size, step=[60, 75, 90], factor=0.1):
    step_ = [epoch_size * (x-begin_epoch) for x in step if x-begin_epoch > 0]
    return mx.lr_scheduler.MultiFactorScheduler(step=step_, factor=factor) if len(step_) else None


def main():        
    if args.data_type == "cifar10":
        args.aug_level = 1
        args.num_classes = 10
        # depth should be one of 110, 164, 1001,...,which is should fit (args.depth-2)%9 == 0
        if((args.depth-2)%9 == 0 and args.depth >= 164):
            per_unit = [(args.depth-2)/9]
            filter_list = [16, 64, 128, 256]
            bottle_neck = True
        elif((args.depth-2)%6 == 0 and args.depth < 164):
            per_unit = [(args.depth-2)/6]
            filter_list = [16, 16, 32, 64]
            bottle_neck = False
        else:
            raise ValueError("no experiments done on detph {}, you can do it youself".format(args.depth))
        units = per_unit*3
        symbol = resnet(units=units, num_stage=3, filter_list=filter_list, num_class=args.num_classes,
                        data_type="cifar10", bottle_neck = bottle_neck, bn_mom=args.bn_mom, workspace=args.workspace,
                        memonger=args.memonger)
    elif args.data_type == "imagenet":
        args.num_classes = 1000
        if args.depth == 18:
            units = [2, 2, 2, 2]
        elif args.depth == 34:
            units = [3, 4, 6, 3]
        elif args.depth == 50:
            units = [3, 4, 6, 3]
        elif args.depth == 101:
            units = [3, 4, 23, 3]
        elif args.depth == 152:
            units = [3, 8, 36, 3]
        elif args.depth == 200:
            units = [3, 24, 36, 3]
        elif args.depth == 269:
            units = [3, 30, 48, 8]
        else:
            raise ValueError("no experiments done on detph {}, you can do it youself".format(args.depth))
        symbol = resnet(units=units, num_stage=4, filter_list=[64, 256, 512, 1024, 2048] if args.depth >=50
                        else [64, 64, 128, 256, 512], num_class=args.num_classes, data_type="imagenet", bottle_neck = True
                        if args.depth >= 50 else False, bn_mom=args.bn_mom, workspace=args.workspace,
                        memonger=args.memonger)
    else:
         raise ValueError("do not support {} yet".format(args.data_type))
    kv = mx.kvstore.create(args.kv_store)
    devs = mx.cpu() if args.gpus is None else [mx.gpu(int(i)) for i in args.gpus.split(',')]
    
    # logging
    head = '%(asctime)-15s Node[' + str(kv.rank) + '] %(message)s'
    if 'log_file' in args and args.log_file is not None:
        log_file = args.log_file
        log_dir = args.log_dir
        log_file_full_name = os.path.join(log_dir, log_file)
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        logger = logging.getLogger()
        handler = logging.FileHandler(log_file_full_name)
        formatter = logging.Formatter(head)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        logger.info('start with arguments %s', args)
    else:
        logging.basicConfig(level=logging.DEBUG, format=head)
        logging.info('start with arguments %s', args)
    
    kv_store_type = ""
    if args.kv_store == "dist_sync":
        kv_store_type = "bsp"
    elif args.kv_store == "dist_async":
        kv_store_type = "asp"
    elif args.kv_store == "dist_gsync":
        kv_store_type = "gsp"
    elif args.kv_store == "dist_ssync":
        kv_store_type = "ssp"
    begin_epoch = args.model_load_epoch if args.model_load_epoch else 0
    user = getpass.getuser()
    if not os.path.exists("/home/{}/mxnet_model/model/{}/resnet{}/{}".format(user, args.data_type, args.depth, kv_store_type)):
        os.makedirs("/home/{}/mxnet_model/model/{}/resnet{}/{}".format(user, args.data_type, args.depth, kv_store_type))
    model_prefix = "/home/{}/mxnet_model/model/{}/resnet{}/{}/{}-{}-resnet{}-{}".format(user, args.data_type, args.depth, kv_store_type, kv_store_type, args.data_type, args.depth, kv.rank)
    checkpoint = None if not args.savemodel else mx.callback.do_checkpoint(model_prefix)

    arg_params = None
    aux_params = None
    if args.retrain:
        _, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, args.model_load_epoch)
    if args.memonger:
        import memonger
        symbol = memonger.search_plan(symbol, data=(args.batch_size, 3, 32, 32) if args.data_type=="cifar10"
                                                    else (args.batch_size, 3, 224, 224))
    
    splits = 1
    part = 0
    val_splits = kv.num_workers
    val_part = kv.rank
    '''yegeyan 2016.10.6'''
    if args.kv_store == "dist_sync" or args.kv_store == "dist_async" or args.kv_store == "dist_ssync":
    #if args.kv_store == "dist_sync":
        splits = kv.num_workers
        part = kv.rank    
    if args.kv_store == "dist_gsync":
        if args.data_allocator == 1:
            if args.hostname == "gpu-cluster-1":
                part = args.cluster1_begin
                splits = args.cluster1_end
            elif args.hostname == "gpu-cluster-2":
                part = args.cluster2_begin
                splits = args.cluster2_end
            elif args.hostname == "gpu-cluster-3":
                part = args.cluster3_begin
                splits = args.cluster3_end
            elif args.hostname == "gpu-cluster-4":
                part = args.cluster4_begin
                splits = args.cluster4_end
            else:
                part = args.cluster5_begin
                splits = args.cluster5_end
    
            args.data_proportion = splits - part
        else:
            splits = kv.num_workers
            part = kv.rank
    
    # yegeyan 2017.1.15
    epoch_size = args.num_examples / args.batch_size
    
    model_args={}
    
    if args.kv_store == 'dist_sync' or args.kv_store == 'dist_async' or args.kv_store == 'dist_ssync':
    #if args.kv_store == 'dist_sync':
        epoch_size /= kv.num_workers
        model_args['epoch_size'] = epoch_size

    '''yegeyan 2016.12.13'''
    if args.kv_store == 'dist_gsync':
        if args.data_allocator == 1:
            epoch_size *= args.data_proportion
            model_args['epoch_size'] = epoch_size
        else:
            epoch_size /= kv.num_workers
            model_args['epoch_size'] = epoch_size

    if 'lr_factor' in args and args.lr_factor < 1:
        model_args['lr_scheduler'] = mx.lr_scheduler.FactorScheduler(
            step=max(int(batch_num * args.lr_factor_epoch), 1),  # yegeyan 2016.12.13
            factor=args.lr_factor)

    if 'clip_gradient' in args and args.clip_gradient is not None:
        model_args['clip_gradient'] = args.clip_gradient
    
    eval_metrics = ['accuracy']
    ## TopKAccuracy only allows top_k > 1
    for top_k in [5, 10, 20]:
        eval_metrics.append(mx.metric.create('top_k_accuracy', top_k=top_k))
    # yegeyan 2017.1.4
    val_eval_metrics = ['accuracy']
    ## TopKAccuracy only allows top_k > 1
    for top_k in [5, 10, 20]:
        val_eval_metrics.append(mx.metric.create('top_k_accuracy', top_k=top_k))

    train = mx.io.ImageRecordIter(
        path_imgrec         = os.path.join(args.data_dir, "train.rec") if args.data_type == 'cifar10' else
                              os.path.join(args.data_dir, "train_480.rec") if args.aug_level == 1
                              else os.path.join(args.data_dir, "train_480.rec"),
        label_width         = 1,
        data_name           = 'data',
        label_name          = 'softmax_label',
        data_shape          = (3, 32, 32) if args.data_type=="cifar10" else (3, 224, 224),
        batch_size          = args.batch_size,
        pad                 = 4 if args.data_type == "cifar10" else 0,
        fill_value          = 127,  # only used when pad is valid
        rand_crop           = True,
        max_random_scale    = 1.0,  # 480 with imagnet, 32 with cifar10
        min_random_scale    = 1.0 if args.data_type == "cifar10" else 1.0 if args.aug_level == 1 else 0.533,  # 256.0/480.0
        max_aspect_ratio    = 0 if args.data_type == "cifar10" else 0 if args.aug_level == 1 else 0.25,
        random_h            = 0 if args.data_type == "cifar10" else 0 if args.aug_level == 1 else 36,  # 0.4*90
        random_s            = 0 if args.data_type == "cifar10" else 0 if args.aug_level == 1 else 50,  # 0.4*127
        random_l            = 0 if args.data_type == "cifar10" else 0 if args.aug_level == 1 else 50,  # 0.4*127
        max_rotate_angle    = 0 if args.aug_level <= 2 else 10,
        max_shear_ratio     = 0 if args.aug_level <= 2 else 0.1,
        rand_mirror         = True,
        shuffle             = True,
        preprocess_threads  = 4,
        num_parts           = splits,
        part_index          = part)
    val = mx.io.ImageRecordIter(
        path_imgrec         = os.path.join(args.data_dir, "test.rec") if args.data_type == 'cifar10' else
                              os.path.join(args.data_dir, "val_480.rec"),
        label_width         = 1,
        data_name           = 'data',
        label_name          = 'softmax_label',
        batch_size          = args.batch_size,
        data_shape          = (3, 32, 32) if args.data_type=="cifar10" else (3, 224, 224),
        rand_crop           = False,
        rand_mirror         = False,
        preprocess_threads  = 4,
        num_parts           = val_splits,
        part_index          = val_part)
    model = mx.model.FeedForward(
        ctx                 = devs,
        symbol              = symbol,
        arg_params          = arg_params,
        aux_params          = aux_params,
        num_epoch           = args.num_epochs,
        begin_epoch         = begin_epoch,
        learning_rate       = args.lr,
        momentum            = args.mom,
        wd                  = args.wd,
        #optimizer           = 'nag',
        optimizer           = 'sgd',
        initializer         = mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2),
        lr_scheduler        = multi_factor_scheduler(begin_epoch, epoch_size, step=[220, 260, 280], factor=0.1)
                             if args.data_type=='cifar10' else
                             multi_factor_scheduler(begin_epoch, epoch_size, step=[30, 60, 90], factor=0.1),
        **model_args
        )
    model.fit(
        X                  = train,
        eval_data          = val,
        eval_metric        = eval_metrics,
        val_eval_metric    = val_eval_metrics,
        kvstore            = kv,
        batch_end_callback = mx.callback.Speedometer(args.batch_size, 50),
        epoch_end_callback = checkpoint, 
        hostname           = socket.gethostbyname_ex(socket.gethostname())[0],
        dataset            = args.data_type,
        staleness          = args.staleness,
        network_name       = "resnet_" + str(args.depth),
        lr                 = args.lr) #yegeyan 2017.5.15
    # logging.info("top-1 and top-5 acc is {}".format(model.score(X = val,
    #               eval_metric = ['acc', mx.metric.create('top_k_accuracy', top_k = 5)])))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="command for training resnet-v2")
    parser.add_argument('--gpus', type=str, default='0', help='the gpus will be used, e.g "0,1,2,3"')
    parser.add_argument('--data-dir', type=str, default='./data/imagenet/', help='the input data directory')
    parser.add_argument('--data-type', type=str, default='imagenet', help='the dataset type')
    parser.add_argument('--list-dir', type=str, default='./',
                        help='the directory which contain the training list file')
    parser.add_argument('--lr', type=float, default=0.1, help='initialization learning reate')
    parser.add_argument('--mom', type=float, default=0.9, help='momentum for sgd')
    parser.add_argument('--bn-mom', type=float, default=0.9, help='momentum for batch normlization')
    parser.add_argument('--wd', type=float, default=0.0001, help='weight decay for sgd')
    parser.add_argument('--batch-size', type=int, default=256, help='the batch size')
    parser.add_argument('--workspace', type=int, default=512, help='memory space size(MB) used in convolution, if xpu '
                        ' memory is oom, then you can try smaller vale, such as --workspace 256')
    parser.add_argument('--depth', type=int, default=50, help='the depth of resnet')
    parser.add_argument('--num-classes', type=int, default=1000, help='the class number of your task')
    parser.add_argument('--aug-level', type=int, default=2, choices=[1, 2, 3],
                        help='level 1: use only random crop and random mirror\n'
                             'level 2: add scale/aspect/hsv augmentation based on level 1\n'
                             'level 3: add rotation/shear augmentation based on level 2')
    parser.add_argument('--num-examples', type=int, default=1281167, help='the number of training examples')
    parser.add_argument('--kv-store', type=str, default='device', help='the kvstore type')
    parser.add_argument('--model-load-epoch', type=int, default=0,
                        help='load the model on an epoch using the model-load-prefix')
    parser.add_argument('--frequent', type=int, default=50, help='frequency of logging')
    parser.add_argument('--memonger', action='store_true', default=False,
                        help='true means using memonger to save momory, https://github.com/dmlc/mxnet-memonger')
    parser.add_argument('--retrain', action='store_true', default=False, help='true means continue training')
    parser.add_argument('--log-file', type=str,
                        help='the name of log file')
    parser.add_argument('--log-dir', type=str, default="output",
                        help='directory of the log file')
    
    parser.add_argument('--num-epochs', type=int, default=1,
                        help='the number of training epochs')
    parser.add_argument('--hostname', type=str, default="gpu-cluster-1",
                        help='the hostname of this worker')
    parser.add_argument('--cluster1-begin', type=float, default=0,
                        help='the begin of data in cluster1')
    parser.add_argument('--cluster1-end', type=float, default=0,
                        help='the end of data in cluster1')
    parser.add_argument('--cluster2-begin', type=float, default=0,
                        help='the begin of data in cluster2')
    parser.add_argument('--cluster2-end', type=float, default=0,
                        help='the end of data in cluster2')
    parser.add_argument('--cluster3-begin', type=float, default=0,
                        help='the begin of data in cluster3')
    parser.add_argument('--cluster3-end', type=float, default=0,
                        help='the end of data in cluster3')
    parser.add_argument('--cluster4-begin', type=float, default=0,
                        help='the begin of data in cluster4')
    parser.add_argument('--cluster4-end', type=float, default=0,
                        help='the end of data in cluster4')
    parser.add_argument('--cluster5-begin', type=float, default=0,
                        help='the begin of data in cluster5')
    parser.add_argument('--cluster5-end', type=float, default=0,
                       help='the end of data in cluster5')
    parser.add_argument('--data_proportion', type=float, default=0,
                       help='the data proportion')
    parser.add_argument('--staleness', type=int, default=0,
                    help='the staleness of dist_ssync')
    parser.add_argument('--savemodel', action='store_true', default=False, 
                    help='true means save model')
    parser.add_argument('--data-allocator', type=int, default=0,
                    help='whether to use data allocator by group')
    args = parser.parse_args()
    main()
