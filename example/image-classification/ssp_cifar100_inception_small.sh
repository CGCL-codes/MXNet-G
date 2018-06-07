python ../../tools/kill-mxnet.py hosts
python ./scp_data_sharding.py
python ./recover_ssp_miniters.py
export PS_VERBOSE=1
../../tools/launch.py -H hosts -n 37 -s 37 -i ib0 --launcher ssh \
python train_cifar100.py --data-dir /home/mxnet_data/cifar100/ \
 --batch-size 20 --lr 0.1 --lr-factor .94 --num-examples 50000 --num-epoch 200 \
 --gpu 0,1 --kv-store dist_ssync --staleness 0 \
# --savemodel \
# --model-load-epoch 1 --retrain
