python ../../tools/kill-mxnet.py hosts
python ./scp_data_sharding.py
python ./recover_ssp_miniters.py
export PS_VERBOSE=1
../../tools/launch.py -H hosts -n 34 -s 34 -i ib0 --launcher ssh -H hosts \
python train_resnet.py --data-dir /home/mxnet_data/cifar10/ \
 --data-type cifar10 --depth 164 --batch-size 20 --num-examples 50000 --num-epoch 300 \
 --gpus 0,1 --kv-store dist_ssync --staleness 0 \
 #--model-load-epoch 219 --retrain
