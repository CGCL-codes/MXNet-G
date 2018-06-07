export PS_VERBOSE=1
../../tools/launch.py -n 10 -s 10 -i ib0 --launcher ssh -H hosts \
 python train_cifar10.py --data-dir /home/mxnet_data/cifar10/ \
 --batch-size 25 --lr 0.1 --lr-factor .94 --num-epoch 1 --num-examples 50000 \
 --gpu 0,1 --kv-store dist_async \
