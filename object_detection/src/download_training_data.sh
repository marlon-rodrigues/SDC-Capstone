#!/bin/bash

PRETRAINED_MODEL=https://s3.amazonaws.com/ricardosdc/capstone_project/ssd_mobilenet_v1_coco_2017_11_17.tar.gz
SDC_DATASET=https://s3.amazonaws.com/ricardosdc/capstone_project/sim_data.record

DESTINATION=$SDC_WORKING_DIR/downloads

wget $PRETRAINED_MODEL -P $DESTINATION
wget $SDC_DATASET -P $DESTINATION

tar -xvz $PRETRAINED_MODEL