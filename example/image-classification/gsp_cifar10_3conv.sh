python ../../tools/kill-mxnet.py hosts
python ./scp_data_sharding.py
python ./scp_groups.py
export PS_VERBOSE=1
../../tools/launch.py -n 40 -s 40 -i ib0 --launcher ssh -H hosts \
 python train_cifar10.py --data-dir /home/mxnet_data/cifar10/ \
 --network small --batch-size 5 --lr 0.001 --lr-factor .94 --num-epoch 5 --num-examples 50000 \
 --gpus 0,1 --kv-store dist_gsync \
