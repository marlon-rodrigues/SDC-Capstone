+ Make all shell scripts executable:

Go to your working folder (`/object_detection_training/`) and then execute
`chmod +x ./src/*.sh`

+ Configure your working folder by editing `/src/start_here.sh`

+ Setup variables, make folders and download training stuff by executing from your working folder:
`. /src/start_here.sh`


+ Read how to install Tensorflow Object Detection API [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md)

When reaching to section *"Add Libraries to PYTHONPATH"* I really encourage to save those variables to your ~/.bashrc , it will avoid many potential mistakes. Trust me, don't waste the same time I wasted on debugging what was going on.

+ Edit *config/ssd_mobilenet_v1_coco_simulator.config* and replace *"/home/ubuntu/SDC/tl_detector/"* with the content of `echo $SDC_WORKING_DIR`

+ Create and activate an environment with the provided `requirements.txt`

+ Execute training:
`./src/execute_training.sh`
