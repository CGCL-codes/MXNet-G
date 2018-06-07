python ../../tools/kill-mxnet.py hosts
python ./scp_data_sharding.py
export PS_VERBOSE=1
../../tools/launch.py -H hosts -n 35 -s 35 -i ib0 --launcher ssh -H hosts \
python train_resnet.py --data-dir /home/mxnet_data/ilsvrc12/ \
 --data-type imagenet --depth 50 --batch-size 40 --num-examples 1281167 --num-epoch 300 \
 --gpus 0,1 --kv-store dist_sync
 #--model-load-epoch 219 --retrain
