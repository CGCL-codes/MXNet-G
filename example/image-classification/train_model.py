import find_mxnet
import mxnet as mx
import logging
import os
import getpass

def fit(args, network, data_loader, batch_end_callback=None):
    # kvstore
    kv = mx.kvstore.create(args.kv_store)

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

    # load model
    kv_store_type = ""
    if args.kv_store == "dist_sync":
        kv_store_type = "bsp"
    elif args.kv_store == "dist_async":
        kv_store_type = "asp"
    elif args.kv_store == "dist_gsync":
        kv_store_type = "gsp"
    elif args.kv_store == "dist_ssync":
        kv_store_type = "ssp"
    user = getpass.getuser()
    network_symbol = None
    if args.network == "inception-bn-28-small":
        network_symbol = "inception_bn_small"
    elif args.network == "inception-bn":
        network_symbol = "inception"
    elif args.network == "3conv":
        network_symbol = "3conv"

    if not os.path.exists("/home/{}/mxnet_model/model/{}/{}/{}".format(user, args.dataset, network_symbol, kv_store_type)):
        os.makedirs("/home/{}/mxnet_model/model/{}/{}/{}".format(user, args.dataset, network_symbol, kv_store_type))

    model_prefix = "/home/{}/mxnet_model/model/{}/{}/{}/{}-{}-{}-{}".format(user, args.dataset, network_symbol, kv_store_type, kv_store_type, args.dataset, network_symbol, kv.rank)

    model_args = {}
    if args.retrain:
        tmp = mx.model.FeedForward.load(model_prefix, args.model_load_epoch)
        model_args = {'arg_params' : tmp.arg_params,
                      'aux_params' : tmp.aux_params,
                      'begin_epoch' : args.model_load_epoch}
        # TODO: check epoch_size for 'dist_sync' yegeyan 2017.1.13
        epoch_size = args.num_examples / args.batch_size
        model_args['begin_num_update'] = epoch_size * args.model_load_epoch
    # save model
    checkpoint = None if not args.savemodel else mx.callback.do_checkpoint(model_prefix)

    # data
    (train, val) = data_loader(args, kv)

    # train
    devs = mx.cpu() if args.gpus is None else [
        mx.gpu(int(i)) for i in args.gpus.split(',')]

    epoch_size = args.num_examples / args.batch_size
    batch_num = args.num_examples / args.batch_size #yegeyan 2016.12.13
    groups_path = '/home/' + getpass.getuser() + '/MXNet-G/example/image-classification/groups'
    group_num = len(open(groups_path, 'rU').readlines()) #yegeyan 2016.12.13
    
    if args.kv_store == 'dist_sync':
        epoch_size /= kv.num_workers
        batch_num /= kv.num_workers
        model_args['epoch_size'] = epoch_size

    if args.kv_store == 'dist_async' or args.kv_store == 'dist_ssync':
        epoch_size /= kv.num_workers
        model_args['epoch_size'] = epoch_size
    
    '''yegeyan 2016.12.13'''
    if  args.kv_store == 'dist_gsync':
        if args.data_allocator == 1:
            epoch_size *= args.data_proportion
            batch_num = batch_num * group_num / kv.num_workers
            model_args['epoch_size'] = epoch_size
        else:
            epoch_size /= kv.num_workers
            #batch_num /= kv.num_workers
            batch_num = batch_num * group_num / kv.num_workers
            model_args['epoch_size'] = epoch_size

    if 'lr_factor' in args and args.lr_factor < 1:
        model_args['lr_scheduler'] = mx.lr_scheduler.FactorScheduler(
            step = max(int(batch_num * args.lr_factor_epoch), 1), #yegeyan 2016.12.13
            factor = args.lr_factor)

    if 'clip_gradient' in args and args.clip_gradient is not None:
        model_args['clip_gradient'] = args.clip_gradient

    # disable kvstore for single device
    if 'local' in kv.type and (
            args.gpus is None or len(args.gpus.split(',')) is 1):
        kv = None

    model = mx.model.FeedForward(
        ctx                = devs,
        symbol             = network,
        num_epoch          = args.num_epochs,
        learning_rate      = args.lr,
        momentum           = 0.9,
        wd                 = 0.00001,
        initializer        = mx.init.Xavier(factor_type="in", magnitude=2.34),
        **model_args)

    eval_metrics = ['accuracy']
    ## TopKAccuracy only allows top_k > 1
    for top_k in [5, 10, 20]:
        eval_metrics.append(mx.metric.create('top_k_accuracy', top_k = top_k))
    #yegeyan 2017.1.4
    val_eval_metrics = ['accuracy']
    ## TopKAccuracy only allows top_k > 1
    for top_k in [5, 10, 20]:
        val_eval_metrics.append(mx.metric.create('top_k_accuracy', top_k = top_k))

    if batch_end_callback is not None:
        if not isinstance(batch_end_callback, list):
            batch_end_callback = [batch_end_callback]
    else:
        batch_end_callback = []
    batch_end_callback.append(mx.callback.Speedometer(args.batch_size, 50))


    model.fit(
        X                  = train,
        eval_data          = val,
        eval_metric        = eval_metrics,
        val_eval_metric    = val_eval_metrics, #yegeyan 2017.1.4
        kvstore            = kv,
        batch_end_callback = batch_end_callback,
        epoch_end_callback = checkpoint,
        hostname           = args.hostname, #yegeyan 2016.10.6
        dataset            = args.dataset,
        staleness          = args.staleness,
        network_name       = args.network,
        lr                 = args.lr) #yegeyan 2017.5.15
