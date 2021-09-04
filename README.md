# se_ssd
SE-SSD is a 3d-object-detection model: https://github.com/Vegeta2020/SE-SSD 


### Usage 

```bash
    $ docker pull loaywael/se_ssd:ros-kinetic-ub16
```

creating a container from this image.
```bash
    $ docker run -itd --name SE-SSD --gpus all --ipc host \
        -v <host-shared-path>:/shared_area loaywael/se_ssd:ros-kinetic-ub16
    $ docker exec -it SE-SSD bash
```
make sure of setting the ros-network between the container and the host
```bash
    $ ifconfig
    $ nano ~/.bashrc
    $ export ROS_MASTER_URI=http://<host-ip>:11311
    $ export ROS_IP=<container-ip>
```
run a demo script 
```bash
    # 1st option using roslaunch
    $ roslaunch se_ssd detect_3d_objects.launch    # default kitti-bag for arl-bag set arl:=true
    
    # 2nd option using python interpreter
    $ cd SE-SSD/tools/
    $ python <script-name>
```
 in case of using rosbag/rviz run them on the host.
<br><br><br>

### The image supports:
- model inference given a directory of PointCloud binaries.
- integrated with ROS
- supports custom datasets

### To Do:
- support multi-class prediction
- support 360 degree detection
