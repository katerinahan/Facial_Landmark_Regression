#!/bin/sh

lst_dir=$1

HOBOT_MXNET=/mnt/hdfs-data-1/data/zhenghua.chen/alphacnn.densebox_vm.20180409
#export LD_LIBRARY_PATH=$HOBOT_MXNET/lib:$LD_LIBRARY_PATH
python $HOBOT_MXNET/tools/im2rec.py --num-thread 8 --color 0 ${lst_dir} .

echo "rec done."