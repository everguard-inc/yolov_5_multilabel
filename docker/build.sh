python ./docker/build.py ${1:-"yolov5"} \
--file ${2:-"./docker/Dockerfile"} \
--cpu-shares $((($(nproc) + 1) / 2))
