#!/bin/bash

# execute like this: . start.sh

# Prepare variables we will use
source ./src/variables.sh

# create required folders
./src/make_folders.sh

# download training data
./src/download_training_data.sh

# download Tensorflow object detection API
git clone https://github.com/tensorflow/models.git $SDC_WORKING_DIR/external_repo/models

