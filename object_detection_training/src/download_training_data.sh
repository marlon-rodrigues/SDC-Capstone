#!/bin/bash

PRETRAINED_MODEL=ssd_mobilenet_v1_coco_2017_11_17.tar.gz
PRETRAINED_MODEL_URL=https://s3.amazonaws.com/ricardosdc/capstone_project/$PRETRAINED_MODEL
SDC_DATASET_URL=https://s3.amazonaws.com/ricardosdc/capstone_project/sim_data.record

DESTINATION=$SDC_WORKING_DIR/downloads

wget $PRETRAINED_MODEL -P $DESTINATION
wget $SDC_DATASET -P $DESTINATION

tar -xvf $DESTINATION/$PRETRAINED_MODEL
