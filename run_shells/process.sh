#!/bin/bash

set -ex

curdir=$(pwd)
echo "curdir:$curdir"
cd "$curdir" || exit

cd ..



#base_dir="/mnt/cephfs/hjh/train_record/tts/bert_vits2"
data_dir="/mnt/cephfs/hjh/train_record/tts/bert_vits2/test_data/"

config_file="`pwd`/configs/config.json"

python run_shells/process/wav2text.py \
--languages ZH \
--whisper_size medium \
--data_dir ${data_dir} \
--config_file ${config_file}