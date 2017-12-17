#!/bin/bash

PATH_TO_PIPELINE_CONFIG=$SDC_WORKING_DIR/config/ssd_mobilenet_v1_coco_simulator.config
PATH_TO_TRAIN_DIR=$SDC_WORKING_DIR/results/train_dir
PATH_TO_EVAL_DIR=$SDC_WORKING_DIR/results/eval_dir

TF_DIR=$SDC_WORKING_DIR/external_repo/models/research

# From the tensorflow/models/research/ directory
cd $TF_DIR

python object_detection/eval.py \
    --logtostderr \
    --pipeline_config_path=$PATH_TO_PIPELINE_CONFIG \
    --checkpoint_dir=$PATH_TO_TRAIN_DIR \
    --eval_dir=$PATH_TO_EVAL_DIR