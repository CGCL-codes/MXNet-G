python ../../tools/kill-mxnet.py hosts
python ./scp_data_sharding.py
export PS_VERBOSE=1
../../tools/launch.py -n 37 -s 35 -i ib0 --launcher ssh -H hosts \
 python train_cifar100.py --network cross --data-dir /home/mxnet_data/cifar100/ \
 --batch-size 20 --lr 0.1 --lr-factor .94 --num-epoch 200 --num-examples 50000 \
 --gpu 0,1 --kv-store dist_async \
