#!/bin/bash

PATH_TO_PIPELINE_CONFIG=$SDC_WORKING_DIR/config/ssd_mobilenet_v1_coco_simulator.config
PATH_TO_CHECKPOINT=$SDC_WORKING_DIR/results/train_dir
PATH_TO_OUTPUT_DIR=$SDC_WORKING_DIR/results/inference_dir

TF_DIR=$SDC_WORKING_DIR/external_repo/models/research

# From the tensorflow/models/research/ directory
cd $TF_DIR

python object_detection/export_inference_graph.py \
    --pipeline_config_path=$PATH_TO_PIPELINE_CONFIG \
    --trained_checkpoint_prefix=$PATH_TO_CHECKPOINT \
    --output_directory=$PATH_TO_OUTPUT_DIR