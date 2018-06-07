#!/bin/bash
./pretrain_1epoch.sh > ./clusting/pretrain.log 2>&1
python ./clusting/aver-speed_auto.py
python ./clusting/clusting_auto.py
python ./clusting/data_sharding.py
