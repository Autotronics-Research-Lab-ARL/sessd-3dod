# Self-Ensembling Single Shot Detector (SESSD)
SE-SSD is a 3d-object-detection model: https://github.com/Vegeta2020/SE-SSD 

<img width="100%" src="assets/sessd_sample.gif">
<p align="center">
    <i>Average inference time</i>: 65[ms] &asymp; 15[fps]
    <i>, benchmark: TITAN GTX, RTX 3060Ti</i><br>
    <i>detection-radius: </i> 50[m] 
</p>

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
##### [on-guest]
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

### [evaluation]
Reproduced the model results
```
Evaluation official_AP_11: car AP(Average Precision)@0.70, 0.70, 0.70:
bbox AP:99.04, 90.06, 89.47
bev  AP:90.59, 88.80, 87.84
3d   AP:90.05, 79.82, 78.80
aos  AP:99.01, 89.81, 89.00
car AP(Average Precision)@0.70, 0.50, 0.50:
bbox AP:99.04, 90.06, 89.47
bev  AP:99.09, 90.23, 89.76
3d   AP:99.05, 90.18, 89.68
aos  AP:99.01, 89.81, 89.00

Evaluation official_AP_40: car AP(Average Precision)@0.70, 0.70, 0.70:
bbox AP:99.63, 93.64, 90.99
bev  AP:96.65, 90.26, 87.58
3d   AP:93.56, 84.14, 81.21
aos  AP:99.60, 93.34, 90.50
car AP(Average Precision)@0.70, 0.50, 0.50:
bbox AP:99.63, 93.64, 90.99
bev  AP:99.67, 96.01, 93.44
3d   AP:99.66, 95.94, 93.35
aos  AP:99.60, 93.34, 90.50
```
### [supports]
- model inference given a directory of PointCloud binaries.
- integrated with ROS
- supports custom datasets
- support 360 degree detection

### [to-do]
- support multi-class prediction

