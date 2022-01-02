# Self-Ensembling Single Shot Detector ([SESSD](https://openaccess.thecvf.com/content/CVPR2021/papers/Zheng_SE-SSD_Self-Ensembling_Single-Stage_Object_Detector_From_Point_Cloud_CVPR_2021_paper.pdf))
SE-SSD is a 3d-object-detection model: https://github.com/Vegeta2020/SE-SSD 

<img width="100%" src="assets/sessd_sample.gif">
<p align="center">
    <i>detection-radius: </i> 50[m],  
    <i>benchmark: TITAN GTX, RTX 3060Ti</i><br>
    <i>Average inference time(180-degree)</i>: 30[ms] &asymp; 30[fps]<br>
    <i>Average inference time(360-degree)</i>: 65[ms] &asymp; 15[fps]<br> 
</p>

### Other Work
<table>
<tr>
    <td width=40%><img width="100%" src="assets/3dod_models.png"></td>
    <td width=60%><img width="100%" src="assets/3dod_table.png"></td>
</tr>
</table>
<br>

### Usage 
ignore the repo src just pull the docker image and follow the usage guide

```bash
$ docker pull loaywael/sessd:ros-torch1.9-cuda111
```


creating a container from this image.
```bash
$ docker run -itd --name SESSD_ROS --gpus all --net host --ipc host \
    -v <host-shared-path>:/shared_area loaywael/sessd:ros-torch1.9-cuda111
$ docker exec -it SESSD_ROS bash
```

run the demo node
```bash
# 1st option using roslaunch
# default kitti-bag for arl-bag set arl:=true
$ roslaunch se_ssd detect_3d_objects.launch kitti:=true weights_path:=<abs-path>
```
<br>

### [evaluation]
Reproduced the model results
```
Evaluation official_AP_11: car AP(Average Precision)@0.70, 0.70, 0.70:
bbox AP:98.72, 90.10, 89.57
bev  AP:90.61, 88.76, 88.18
3d   AP:90.21, 86.25, 79.22
aos  AP:98.67, 89.86, 89.16
car AP(Average Precision)@0.70, 0.50, 0.50:
bbox AP:98.72, 90.10, 89.57
bev  AP:98.76, 90.19, 89.77
3d   AP:98.73, 90.16, 89.72
aos  AP:98.67, 89.86, 89.16

Evaluation official_AP_40: car AP(Average Precision)@0.70, 0.70, 0.70:
bbox AP:99.57, 95.58, 93.16
bev  AP:96.70, 92.15, 89.75
3d   AP:93.75, 86.18, 83.51
aos  AP:99.52, 95.28, 92.69
car AP(Average Precision)@0.70, 0.50, 0.50:
bbox AP:99.57, 95.58, 93.16
bev  AP:99.60, 95.92, 93.42
3d   AP:99.59, 95.86, 93.36
aos  AP:99.52, 95.28, 92.69
```

### [to-do]
- [x] model inference script.
- [x] integrate with ROS
- [x] support custom datasets
- [x] support 360 degree detection
- [ ] support multi-class prediction

