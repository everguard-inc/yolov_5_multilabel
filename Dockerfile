# YOLOv5 🚀 by Ultralytics, GPL-3.0 license

# Start FROM Nvidia PyTorch image https://ngc.nvidia.com/catalog/containers/nvidia:pytorch
FROM nvcr.io/nvidia/pytorch:21.10-py3

# Install linux packages
RUN apt update && apt install -y zip htop screen libgl1-mesa-glx

# eg_data_tools
COPY eg_data_tools eg_data_tools/
RUN pip install -e eg_data_tools

# Install python dependencies
COPY requirements.txt .
RUN python3 -m pip install --upgrade pip --no-cache
RUN pip uninstall -y nvidia-tensorboard nvidia-tensorboard-plugin-dlprof
RUN pip install -r requirements.txt --no-cache
RUN pip install "opencv-python-headless<=4.3" --no-cache

# Create a user who will own the files created in the docker container 
ARG USER_ID
ARG GROUP_ID

RUN addgroup --gid $GROUP_ID appgroup 
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID appuser

# Create working directory
RUN mkdir -p /app
WORKDIR /app

# Copy contents
COPY . /app

# Change workdir permissions
RUN chown -R appuser:appgroup /app && chmod 755 /app

# Change user from root to appuser
USER appuser

# Downloads to user config dir
ADD https://ultralytics.com/assets/Arial.ttf /root/.config/Ultralytics/


# Set environment variables
# ENV HOME=/usr/src/app


# Usage Examples -------------------------------------------------------------------------------------------------------

# Build and Push
# t=ultralytics/yolov5:latest && sudo docker build -t $t . && sudo docker push $t

# Pull and Run
# t=ultralytics/yolov5:latest && sudo docker pull $t && sudo docker run -it --ipc=host --gpus all $t

# Pull and Run with local directory access
# t=ultralytics/yolov5:latest && sudo docker pull $t && sudo docker run -it --ipc=host --gpus all -v "$(pwd)"/datasets:/usr/src/datasets $t

# Kill all
# sudo docker kill $(sudo docker ps -q)

# Kill all image-based
# sudo docker kill $(sudo docker ps -qa --filter ancestor=ultralytics/yolov5:latest)

# Bash into running container
# sudo docker exec -it 5a9b5863d93d bash

# Bash into stopped container
# id=$(sudo docker ps -qa) && sudo docker start $id && sudo docker exec -it $id bash

# Clean up
# docker system prune -a --volumes

# Update Ubuntu drivers
# https://www.maketecheasier.com/install-nvidia-drivers-ubuntu/

# DDP test
# python -m torch.distributed.run --nproc_per_node 2 --master_port 1 train.py --epochs 3

# GCP VM from Image
# docker.io/ultralytics/yolov5:latest
