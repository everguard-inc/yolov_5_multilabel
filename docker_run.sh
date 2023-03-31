docker run \
--rm -it \
--gpus all \
--shm-size 8G \
--workdir $(pwd) \
--user $(id -u):$(id -g) \
--mount type=bind,source=$HOME,target=$HOME \
$USER/yolov5
