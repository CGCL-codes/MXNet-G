python ../../tools/kill-mxnet.py hosts
python ./scp_data_sharding.py
export PS_VERBOSE=1
../../tools/launch.py -H hosts -n 40 -s 40 -i ib0 --launcher ssh \
python train_cifar10.py --data-dir /home/mxnet_data/cifar10/ \
 --network 3conv --batch-size 5 --lr 0.001 --lr-factor .94 --num-examples 50000 --num-epoch 5 \
 --gpu 0,1 --kv-store dist_async \
 --savemodel \
