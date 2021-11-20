XSOCK=/tmp/.X11-unix
XAUTH=/tmp/.docker.xauth
touch $XAUTH

xauth nlist $DISPLAY | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge -

DISPLAY_NUMBER=$(echo $DISPLAY | cut -d. -f1 | cut -d: -f2)
CONTAINER_DISPLAY=:${DISPLAY_NUMBER}

ARG1=${1:-0}
ARG2=${2:-9999}
NV_GPU=$ARG1

docker run \
--rm -it \
--gpus all \
--shm-size 8G \
-p $ARG2:$ARG2 \
--env="QT_X11_NO_MITSHM=1" \
--env="DISPLAY=${CONTAINER_DISPLAY}" \
--env="XAUTHORITY=${XAUTH}" \
--volume=$XSOCK:$XSOCK:rw \
--volume=$XAUTH:$XAUTH:rw \
--hostname $(hostname) \
--mount type=bind,source="$PWD",target=/app \
--mount type=bind,source="/dev",target=/dev \
--privileged \
$(whoami)/${1:-"yolov5"}
