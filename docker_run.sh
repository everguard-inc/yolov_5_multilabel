sudo docker run \
--rm -it \
--gpus all \
--shm-size 8G \
--hostname $(hostname) \
--mount type=bind,source="$PWD",target=/app \
--mount type=bind,source="/home",target=/home \
--mount type=bind,source="/media",target=/media \
--privileged \
vv/multilabel_ppe_detection