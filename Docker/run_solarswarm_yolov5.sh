docker run -it \
    --name=jetson-yolov5-test \
    --network=host \
    -v /home/$USER/Development/Volumes/noetic-pytorch/volume:/home/volume/ \
    --runtime nvidia \
    solarswarm/yolov5