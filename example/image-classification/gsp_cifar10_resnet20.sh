python ../../tools/kill-mxnet.py hosts
python ./scp_data_sharding.py
python ./scp_groups.py
export PS_VERBOSE=1
../../tools/launch.py -H hosts -n 5 -s 5 -i eth1 --launcher ssh -H hosts \
 python train_resnet.py --data-dir /home/yegeyan/mxnet_data/cifar10/ --data-type cifar10 \
 --depth 20 --lr 0.05 --batch-size 25 --num-epoch 1 --num-examples 50000 \
 --gpus 0,1 --kv-store dist_gsync \
 --savemodel \
 #--model-load-epoch 0 --retrain
