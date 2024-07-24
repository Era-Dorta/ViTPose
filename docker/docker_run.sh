docker run -it \
       --rm \
       --shm-size=2gb \
       --gpus all -e DISPLAY=:1 \
       -v /tmp/.X11-unix:/tmp/.X11-unix \
       -v $(pwd):/workspace \
       local/vitpose:mim \
       /bin/bash