python ../../tools/kill-mxnet.py hosts
python ./scp_data_sharding.py
export PS_VERBOSE=1
../../tools/launch.py -H hosts -n 5 -s 5 -i eth1 --launcher ssh -H hosts \
python train_resnet.py --data-dir /home/yegeyan/mxnet_data/ilsvrc12/ \
 --data-type imagenet --depth 34 --batch-size 25 --num-epoch 1 --num-examples 1281167 \
 --gpus 0,1 --kv-store dist_sync
