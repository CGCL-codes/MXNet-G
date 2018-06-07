python ../../tools/kill-mxnet.py hosts
python ./scp_data_sharding.py
export PS_VERBOSE=1
../../tools/launch.py -H hosts -n 37 -s 37 -i ib0 --launcher ssh -H hosts \
python train_resnet.py --data-dir /home/mxnet_data/cifar10/ \
 --data-type cifar10 --depth 164 --batch-size 20 --num-examples 50000 --num-epoch 300 \
 --gpus 0,1 --kv-store dist_sync
 #--model-load-epoch 219 --retrain
