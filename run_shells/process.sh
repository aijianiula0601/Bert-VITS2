#!/bin/bash

set -ex

curdir=$(pwd)
echo "curdir:$curdir"
cd "$curdir" || exit

cd ..



base_dir="/mnt/cephfs/hjh/train_record/tts/bert_vits2/test_data"
data_dir="${base_dir}/vctk/"
#data_dir="${base_dir}/debug_data/"

save_meta_file="${base_dir}/vctk_meta.txt"
save_vocab_file="${base_dir}/vctk_vocab.txt"
config_file="`pwd`/configs/config.json"


rm -rf ${save_meta_file}
rm -rf ${save_vocab_file}

python run_shells/process/wav2text.py \
--languages EN \
--whisper_size medium \
--data_dir ${data_dir} \
--config_file ${config_file} \
--save_meta_file ${save_meta_file} \
--save_vocab_file ${save_vocab_file}