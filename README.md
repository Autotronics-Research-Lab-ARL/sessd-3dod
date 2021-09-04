# se_ssd
SE-SSD is a 3d-object-detection model: https://github.com/Vegeta2020/SE-SSD 


### Usage 
ignore the repo src just pull the docker image and follow the usage guide

```bash
$ docker pull loaywael/se_ssd:ros-kinetic-ub16
```

##### [on-host]
prepare rosbags/ on current directory
```
./
├── rosbags/
│   ├── kitti_2011_09_26_drive_0020_synced.bag
│   └── raw_data_bag.bag
│
├── bag_streamer.launch
├── rviz_arl_config.rviz
└── rviz_kitti_config.rviz
```
setup ros-network in .bashrc
```bash
$ ifconfig
$ nano ~/.bashrc
$ export ROS_MASTER_URI=http://<host-ip>:11311
$ export ROS_IP=<host-ip>
```
run the bag and rviz
```bash
# default kitti-bag for arl-bag set arl:=true
$ roslaunch ~/shared_pool/3dod/se_ssd/bag_streamer.launch
$ rviz    # load the rviz_config.rviz
```

creating a container from this image.
```bash
$ docker run -itd --name SE-SSD --gpus all --ipc host \
    -v <host-shared-path>:/shared_area loaywael/se_ssd:ros-kinetic-ub16
$ docker exec -it SE-SSD bash
```
##### [on-local]
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
# default kitti-bag for arl-bag set arl:=true
$ roslaunch se_ssd detect_3d_objects.launch

# 2nd option using python interpreter
$ cd SE-SSD/tools/
$ python <script-name>
```
<br>

### [supports]:
- model inference given a directory of PointCloud binaries.
- integrated with ROS
- supports custom datasets

### [to-do]:
- support multi-class prediction
- support 360 degree detection
