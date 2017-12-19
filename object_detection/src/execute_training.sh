#!/bin/bash

PATH_TO_PIPELINE_CONFIG=$SDC_WORKING_DIR/config/ssd_mobilenet_v1_coco_simulator.config
PATH_TO_TRAIN_DIR=$SDC_WORKING_DIR/results/train_dir

TF_DIR=$SDC_WORKING_DIR/external_repo/models/research

# From the tensorflow/models/research/ directory
cd $TF_DIR

python object_detection/train.py \
    --logtostderr \
    --pipeline_config_path=$PATH_TO_PIPELINE_CONFIG \
    --train_dir=$PATH_TO_TRAIN_DIR