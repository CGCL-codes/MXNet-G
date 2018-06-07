python ../../tools/kill-mxnet.py hosts
python ./scp_data_sharding.py
export PS_VERBOSE=1;
../../tools/launch.py -n 1 -s 1 -i ib0 --launcher ssh -H hosts \
 python train_imagenet.py --data-dir /home/mxnet_data/ilsvrc12/ \
 --network inception-bn --batch-size 30 --lr 0.5 --lr-factor .98 --num-epoch 200 \
 --gpus 0,1 --kv-store dist_sync \
 #--savemodel
